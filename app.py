import os
import sqlite3
import pandas as pd
import streamlit as st
import dotenv

dotenv.load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Text to SQL Chatbot", layout="wide")
st.title("📊 Text-to-SQL Chatbot (Excel)")
st.caption("Ask questions in English and query your Excel data")

# -------------------------------
# Load Excel → SQLite (cached)
# -------------------------------
@st.cache_resource
def load_database():
    df = pd.read_excel("customers.xlsx", sheet_name="Table1")
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql("customers", conn, index=False, if_exists="replace")
    return df, conn

df, conn = load_database()

# -------------------------------
# Schema RAG
# -------------------------------
schema_text = f"""
Table: customers
Columns:
{', '.join(df.columns)}
"""

docs = [Document(page_content=schema_text)]

@st.cache_resource
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(docs, embeddings)

vectordb = load_vectordb()

# -------------------------------
# LLM (Hugging Face)
# -------------------------------
@st.cache_resource
def load_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="conversational",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        max_new_tokens=256,
        temperature=0
    )
    return ChatHuggingFace(llm=endpoint)

llm = load_llm()

# -------------------------------
# Prompt
# -------------------------------
prompt = ChatPromptTemplate.from_template("""
You are an expert SQL assistant.

Schema:
{schema}

Rules:
- Use ONLY the given columns
- Do NOT invent columns
- Output ONLY valid SQLite SQL
- No explanation
- No markdown

Question:
{question}
""")

parser = StrOutputParser()

# -------------------------------
# Chat UI
# -------------------------------
question = st.text_input("Ask a question about the customers data:")

if st.button("Run Query"):
    if not question.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            try:
                schema = vectordb.similarity_search(question, k=1)[0].page_content
                chain = prompt | llm | parser

                sql_query = chain.invoke({
                    "schema": schema,
                    "question": question
                })

                if not sql_query or not sql_query.strip():
                    raise ValueError("LLM returned empty SQL")

                sql_query = (
                    sql_query
                    .replace("```sql", "")
                    .replace("```", "")
                    .strip()
                )

                result_df = pd.read_sql_query(sql_query, conn)

                st.subheader("🧠 Generated SQL")
                st.code(sql_query, language="sql")

                st.subheader("📊 Query Result")
                st.dataframe(result_df)

            except Exception as e:
                st.error(f"ERROR: {str(e)}")
