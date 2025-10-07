import os
import sys
import streamlit as st
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from google import genai
from google.genai import types
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from datetime import datetime
import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import NotFound, Conflict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config as cfg


# -----------------------------------------------------------
# INIT
# -----------------------------------------------------------
load_dotenv()
console = Console()

st.set_page_config(page_title="RAG Explorer", layout="wide")


# Initialize clients
@st.cache_resource
def init_qdrant():
    return QdrantClient(url=cfg.QDRANT_URL, api_key=cfg.QDRANT_API_KEY, timeout=60.0)


@st.cache_resource
def init_embedder():
    return TextEmbedding(model_name=cfg.EMBED_MODEL, provider="torch", device="cuda")


@st.cache_resource
def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def load_csv(client, table_ref):
    try:
        df_existing = client.list_rows(table_ref).to_dataframe()
    except NotFound:
        df_existing = pd.DataFrame()
    return df_existing


@st.cache_resource
def init_bigquery():
    client = bigquery.Client()
    dataset_id = f"{client.project}.feedback_dt"

    try:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        client.create_dataset(dataset)
    except Conflict:
        pass
    return client, dataset_id


def upload_csv(client, df, table_ref):
    job = client.load_table_from_dataframe(
        df,
        table_ref,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    job.result()
    return


client_qdrant = init_qdrant()
embedder = init_embedder()
client_genai = init_gemini()

client_bg, dataset_id = init_bigquery()

table_id = "feedback"
table_ref = f"{dataset_id}.{table_id}"
df = load_csv(client_bg, table_ref)


# -----------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------
def search_papers(query, categories=None, top_k=5):
    must_filters = []
    if categories:
        for cat in categories:
            must_filters.append(
                models.FieldCondition(
                    key="categories", match=models.MatchText(text=cat)
                )
            )

    results = client_qdrant.query_points(
        collection_name=cfg.COLLECTION,
        query=models.Document(text=query, model=cfg.EMBED_MODEL),
        limit=top_k,
        query_filter=models.Filter(must=must_filters if categories else None),
    )
    return results


def query_gemini(prompt: str):
    generation_config = types.GenerateContentConfig(temperature=0.3)
    response = client_genai.models.generate_content(
        model=cfg.GEMINI_MODEL, contents=prompt, config=generation_config
    )
    return response.text


def build_prompt(query, retrieved_docs):
    context = "\n\n".join(
        [
            f"Title: {doc.payload['title']}\n"
            f"Abstract: {doc.payload['abstract_chunk']}\n"
            f"Authors: {doc.payload['authors']}\n"
            f"URL: {doc.payload['url']}\n"
            f"Published date: {doc.payload['published']}"
            for doc in retrieved_docs.points
        ]
    )

    prompt_base = """You are a research assistant.
Your goal is to provide a clear and accurate explanation using only the provided context from the papers.
Do not copy phrases like "From the abstract...".
Instead, synthesize the information into a natural explanation.
Do not speculate or hallucinate.
Whenever possible, enrich your answer with insights that appear in the papers.

At the end, provide a section titled "References" with:
- **<title>**
    - Authors: <authors>
    - Published date: <date>
    - URL: <url>

Question: {query}

Context:
{context}

Output must start with "Answer:..."
"""
    return prompt_base.format(query=query, context=context)


def rag_pipeline(query, categories):
    docs = search_papers(query, categories=categories, top_k=10)
    prompt = build_prompt(query, docs)
    response = query_gemini(prompt)
    return response, docs


# -----------------------------------------------------------
# UI
# -----------------------------------------------------------
st.title("🔍 RAG Research Assistant")
st.markdown(
    "Ask a question and explore retrieved academic papers from Arxiv using Gemini + Qdrant."
)

query = st.text_input("Enter your question:", placeholder="e.g. What is a RAG system?")
categories_input = st.text_input(
    "Filter by categories (comma-separated, optional):", placeholder="e.g. cs.CL, cs.AI"
)

answer_raw = None

if st.button("Search", type="primary"):
    if query.strip():
        categories = (
            [c.strip() for c in categories_input.split(",")]
            if categories_input
            else None
        )
        with st.spinner("Retrieving and generating answer..."):
            answer_raw, retrieved_docs = rag_pipeline(query, categories)

        text = "Answer:"
        if answer_raw[:7] == "Answer:":
            answer_raw = answer_raw[7:].strip()

        if "References:" in answer_raw:
            answer = answer_raw.split("References:")[0]

        st.session_state.answer_raw = answer_raw
        st.subheader("🧠 Gemini answer")
        st.markdown(answer)

        st.subheader("📚 Retrieved papers")
        for doc in retrieved_docs.points:
            st.markdown(
                f"**{doc.payload['title']}**  \n"
                f"Authors: {doc.payload['authors']}  \n"
                f"URL: [{doc.payload['url']}]({doc.payload['url']})  \n"
                f"Published: {doc.payload['published']}"
            )

            pdf_url = doc.payload["url"].replace("abs", "pdf")

            pdf_viewer = f"""
            <iframe src="{pdf_url}" width="100%" height="600px"></iframe>
            <p style='font-size:12px;color:gray;'>Papers retrieved from <a href='https://arxiv.org' target='_blank'>arXiv.org</a></p>
            """
            with st.expander("📄 View PDF"):
                st.markdown(pdf_viewer, unsafe_allow_html=True)

            st.write("---")
    else:
        st.warning("Please enter a question to start.")


with st.form("feedback_form"):
    st.subheader("💬 Feedback")

    usefulness = st.radio(
        "How useful was this answer?",
        ["Very useful 👍", "Useful 🙂", "Somewhat useful 😐", "Not useful 👎"],
        horizontal=True,
    )

    relevance = st.slider(
        "How relevant was the answer to your question?",
        1,
        5,
        4,
        help="1 = Not relevant, 5 = Perfectly relevant",
    )

    clarity = st.slider(
        "How clear was the explanation?",
        1,
        5,
        4,
        help="1 = Very confusing, 5 = Very clear",
    )

    detail_level = st.radio(
        "How would you rate the level of detail?",
        ["Too general", "About right", "Too technical"],
    )

    comments = st.text_area(
        "Additional comments or suggestions:", placeholder="Write here..."
    )

    submitted = st.form_submit_button("Submit feedback", disabled=(answer_raw is None))

    if submitted:
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": query,
            "answer": st.session_state.get("answer_raw", ""),
            "usefulness": usefulness,
            "relevance": relevance,
            "clarity": clarity,
            "detail_level": detail_level,
            "comments": comments,
        }

        df = pd.concat([df, pd.DataFrame([feedback_entry])], ignore_index=True)

        upload_csv(client_bg, df, table_ref)

        st.markdown(" Thank you for your feedback! ✅")
