import pandas as pd
import os
import sys
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from google import genai
from google.genai import types
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config as cfg

load_dotenv()


def init_qdrant_client():
    client = QdrantClient(url=cfg.QDRANT_URL, api_key=cfg.QDRANT_API_KEY, timeout=60.0)
    return client


def search_papers(query, collection_name="articles-rag-cos", categories=None, top_k=5):
    must_filters = []
    if categories:
        for cat in categories:
            must_filters.append(
                models.FieldCondition(
                    key="categories", match=models.MatchText(text=cat)
                )
            )

    results = client_qdrant.query_points(
        collection_name=collection_name,
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
    Your goal is to provide a clear and accurate explanation in your own words,
    using only the provided context from the papers as background knowledge.
    Do not copy phrases like "From the abstract...".
    Instead, synthesize the information into a natural explanation.
    Do not speculate or hallucinate.
    Whenever possible, enrich your answer with relevant insights that appear in the papers.

    At the end, provide a section titled "References" with a bullet-point list. Each item must contain exactly these fields (in this order):
    - ** <title> **
        - Authors: <authors>
        - Published date: <date or "Not stated in provided context">
        - URL: <url or "Not stated in provided context">

    Question: {query}

    Context:
    {context}

    Output must start with "Answer:..."
    """

    prompt = prompt_base.format(query=query, context=context)
    return prompt


def rag_pipeline(query, categories):
    docs = search_papers(
        query, collection_name=cfg.COLLECTION, categories=categories, top_k=20
    )
    prompt = build_prompt(query, docs)
    response = query_gemini(prompt)
    return response


if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    client_genai = genai.Client(api_key=api_key)

    embed_model = TextEmbedding(
        model_name=cfg.EMBED_MODEL, provider="torch", device="cuda"
    )
    client_qdrant = init_qdrant_client()

    console = Console()

    # Example
    query = "What is a RAG system?"
    categories = ["cs.CL"]

    response = rag_pipeline(query, categories)

    md = Markdown(response)
    console.print(md)
