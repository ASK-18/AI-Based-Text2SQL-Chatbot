import os
import re
import sqlite3
import pandas as pd
import streamlit as st
import dotenv
from difflib import get_close_matches

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

# ======================================================
# 🔧 Utilities: Column + Table Fixers
# ======================================================

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def build_column_map(columns):
    return {normalize(col): col for col in columns}


def fix_sql_columns(sql: str, column_map: dict):
    """
    Fix column names:
    - spaces
    - underscores
    - numeric-only names
    - avoid double quoting
    """
    tokens = re.findall(r'(?<!")\b[A-Za-z_][A-Za-z0-9_]*\b|\b\d+\b', sql)

    for token in tokens:
        norm = normalize(token)
        match = get_close_matches(norm, column_map.keys(), n=1, cutoff=0.75)
        if match:
            real_col = column_map[match[0]]
            sql = re.sub(
                rf'(?<!")\b{re.escape(token)}\b(?!")',
                f'"{real_col}"',
                sql
            )
    return sql


def fix_table_name(sql: str, table_name: str = "data"):
    """
    Force correct table name.
    Rewrites any FROM <something> to FROM data
    """
    return re.sub(
        r'FROM\s+["\']?[A-Za-z_][A-Za-z0-9_]*["\']?',
        f'FROM {table_name}',
        sql,
        flags=re.IGNORECASE
    )

# ======================================================
# 🎨 Streamlit UI
# ======================================================

st.set_page_config(page_title="Excel → SQL Chatbot", layout="wide")
st.title("📊 Excel → SQL Chatbot")
st.caption("Upload an Excel file, ask questions in English, get SQL + answers")

uploaded_file = st.file_uploader(
    "Upload an Excel file (.xlsx)",
    type=["xlsx"]
)

# ======================================================
# 🧠 Load LLM (Cached)
# ======================================================

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

# ======================================================
# 🧾 Prompt
# ======================================================

prompt = ChatPromptTemplate.from_template("""
You are an expert SQLite SQL assistant.

CRITICAL RULES:
- There is ONLY ONE table named: data
- You MUST always use: FROM data
- NEVER use column names as table names
- NEVER invent table names

Schema:
{schema}

OTHER RULES:
- Use ONLY the given columns
- Column names may contain spaces or numbers
- NEVER rename columns
- Output ONLY valid SQLite SQL
- No explanation
- No markdown

Question:
{question}
""")

parser = StrOutputParser()

# ======================================================
# 🚀 Main Logic
# ======================================================

if uploaded_file is not None:
    try:
        # ----------------------------
        # Read Excel
        # ----------------------------
        df = pd.read_excel(uploaded_file)
        st.success("Excel file loaded successfully")

        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        # ----------------------------
        # SQLite (in-memory)
        # ----------------------------
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        table_name = "data"
        df.to_sql(table_name, conn, index=False, if_exists="replace")

        # ----------------------------
        # Schema + RAG
        # ----------------------------
        schema_text = f"""
        Table: {table_name}

        Columns (use EXACT names):
        {chr(10).join([f'- "{col}"' for col in df.columns])}
        """

        docs = [Document(page_content=schema_text)]

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectordb = FAISS.from_documents(docs, embeddings)

        column_map = build_column_map(df.columns.tolist())

        # ----------------------------
        # Ask Question
        # ----------------------------
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

                # 🔥 Fix columns
                sql_query = fix_sql_columns(sql_query, column_map)

                # 🔥 Force correct table name
                sql_query = fix_table_name(sql_query, table_name="data")

                # Execute
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
