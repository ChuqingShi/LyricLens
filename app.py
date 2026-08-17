import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.embeddings.embedder import Embedder
from src.feedback import ensure_feedback_table, log_feedback
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
    conn = connect_db()
    conn.autocommit = True
    ensure_feedback_table(conn)
    return conn


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

    st.session_state.search_id = st.session_state.get("search_id", 0) + 1
    st.session_state.query = query
    st.session_state.recommendations = recommendations

if "recommendations" in st.session_state:
    conn = get_db_connection()
    search_id = st.session_state.search_id
    result_query = st.session_state.query
    recommendations = st.session_state.recommendations

    cols = st.columns(len(recommendations))
    for i, (col, rec) in enumerate(zip(cols, recommendations), start=1):
        with col, st.container(border=True):
            st.subheader(f"{i}. {rec.title} — {rec.performer}")
            st.write(rec.lyric_scene)
            st.caption(rec.reason)

            feedback_key = f"feedback_{search_id}_{i}"
            logged_key = f"{feedback_key}_logged"
            rating = st.feedback("thumbs", key=feedback_key)
            if rating is not None and st.session_state.get(logged_key) != rating:
                log_feedback(conn, result_query, rec.title, rec.performer, rating)
                st.session_state[logged_key] = rating
