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
st.set_page_config(page_title="Excel to SQL Chatbot", layout="wide")
st.title("📊 Excel → SQL Chatbot")
st.caption("Upload an Excel file, ask questions in English, get SQL + answers")

# -------------------------------
# Upload Excel
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an Excel file (.xlsx)",
    type=["xlsx"]
)

# -------------------------------
# Load LLM (cached)
# -------------------------------
@st.cache_resource
def load_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
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
# Main Logic
# -------------------------------
if uploaded_file is not None:
    try:
        # Read Excel
        df = pd.read_excel(uploaded_file)
        st.success("Excel file loaded successfully")

        st.subheader("📄 Preview of Data")
        st.dataframe(df.head())

        # SQLite (in-memory)
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        table_name = "data"
        df.to_sql(table_name, conn, index=False, if_exists="replace")

        # Schema
        schema_text = f"""
        Table: {table_name}
        Columns:
        {', '.join(df.columns)}
        """

        docs = [Document(page_content=schema_text)]

        # Vector DB (cached per upload)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectordb = FAISS.from_documents(docs, embeddings)

        # Ask question
        question = st.text_input("Ask a question about this data:")

        if st.button("Run Query") and question.strip():
            with st.spinner("Thinking..."):
                schema = vectordb.similarity_search(question, k=1)[0].page_content
                chain = prompt | llm | parser

                sql_query = chain.invoke({
                    "schema": schema,
                    "question": question
                })

                if not sql_query or not sql_query.strip():
                    raise ValueError("LLM returned empty SQL")

                # Clean SQL
                sql_query = (
                    sql_query
                    .replace("```sql", "")
                    .replace("```", "")
                    .strip()
                )

                # Execute SQL
                result_df = pd.read_sql_query(sql_query, conn)

                # Display
                st.subheader("🧠 Generated SQL")
                st.code(sql_query, language="sql")

                st.subheader("📊 Query Result")
                st.dataframe(result_df)

    except Exception as e:
        st.error(f"ERROR: {str(e)}")

else:
    st.info("👆 Upload an Excel file to get started")
