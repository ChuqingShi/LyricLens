import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.embeddings.embedder import Embedder
from src.ingest_postgres import connect_db
from src.rag import RAGPgVector

load_dotenv()

st.set_page_config(page_title="LyricLens", page_icon="🎵")


@st.cache_resource
def get_embedder():
    return Embedder()


@st.cache_resource
def get_openai_client():
    return OpenAI()


@st.cache_resource
def get_db_connection():
    return connect_db()


st.title("🎵 LyricLens")
st.caption(
    "Describe your mood, feelings, occasion, or vibe. "
    "LyricLens finds Billboard Hot 100 songs whose lyrics match what you're looking for."
)

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart the app.")
    st.stop()

query = st.text_input(
    "How are you feeling, and what do you want to feel?",
    placeholder="e.g. stressed about a deadline, need a peaceful breakup song...",
)

if st.button("Find songs", type="primary") and query:
    try:
        embedder = get_embedder()
        openai_client = get_openai_client()
        conn = get_db_connection()
    except FileNotFoundError:
        st.error("Embedding model not found. Run `python -m src.embeddings.download` first.")
        st.stop()

    assistant = RAGPgVector(conn=conn, embedder=embedder, llm_client=openai_client)

    with st.spinner("Searching lyrics and picking your songs..."):
        try:
            recommendations = assistant.rag(query)
        except RuntimeError as e:
            st.error(f"{e} Run the ingest step first: `python -m src.ingest_postgres`.")
            st.stop()

    for i, rec in enumerate(recommendations, start=1):
        with st.container(border=True):
            st.subheader(f"{i}. {rec.title} — {rec.performer}")
            st.write(rec.lyric_scene)
            st.caption(rec.reason)
