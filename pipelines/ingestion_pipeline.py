import os
import sys
import arxiv
from arxiv import UnexpectedEmptyPageError
from datetime import datetime, timedelta
import pandas as pd
import re
import unicodedata
from transformers import AutoTokenizer
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from prefect import flow, task

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config as cfg


def fetch_arxiv(query: str):
    search = arxiv.Search(
        query=query,
        max_results=cfg.ARXIV_MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    return search


@task
def init_arxiv_client():
    client = arxiv.Client(
        page_size=cfg.ARXIV_PAGE_SIZE,
        delay_seconds=cfg.ARXIV_DELAY,
        num_retries=cfg.ARXIV_RETRIES,
    )
    return client


@task
def download_arxiv_articles(
    years: int, categories: list, keywords: list
) -> pd.DataFrame:
    client = init_arxiv_client()
    now = datetime.now()
    start_date = (now - timedelta(days=365 * years)).strftime("%Y%m%d%H%M")
    end_date = now.strftime("%Y%m%d%H%M")

    articles = []
    for cat, name in categories:
        for keyword in keywords:
            query = f'cat:{cat} AND ti:"{keyword}" AND submittedDate:[{start_date} TO {end_date}]'
            search = fetch_arxiv(query=query)
            try:
                for result in client.results(search):
                    articles.append(
                        {
                            "title": result.title,
                            "categories": result.categories,
                            "keyword": keyword,
                            "abstract": result.summary,
                            "authors": [author.name for author in result.authors],
                            "url": result.entry_id,
                            "published": result.published,
                            "year": result.published.year,
                        }
                    )
            except UnexpectedEmptyPageError:
                continue

    df = pd.DataFrame(data=articles)
    return df


@task
def clean_data(df):
    idx_duplicated = df[df[["title", "abstract", "url"]].duplicated()].index
    df.drop(idx_duplicated, axis=0, inplace=True)

    df["categories"] = df["categories"].apply(lambda x: ", ".join(x))
    df["year"] = df["year"].astype(str)
    df["published"] = df["published"].astype(str)
    df["authors"] = df["authors"].apply(lambda x: ", ".join(x))

    short_abstracts = df[df.abstract.apply(lambda x: len(x)) < 200].index
    df.drop(short_abstracts, axis=0, inplace=True)

    nan_titles = df[df.title.isna()].index
    df.drop(nan_titles, axis=0, inplace=True)

    return df


@task
def save_data(df, path, filename):
    df.to_parquet(path + filename, engine="pyarrow", index=False)
    return


def clean_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_latex(text: str) -> str:
    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"\{.*?\}", " ", text)
    return text


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\S+", "", text)


def remove_special_chars(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s.,;:!?()\-']", "", text)


def remove_emails(text: str) -> str:
    return re.sub(r"\S+@\S+", "", text)


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_latex(text)
    text = remove_special_chars(text)
    text = clean_whitespace(text)
    return text


def chunk_by_tokens(text, tokenizer, chunk_size=512, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i : i + chunk_size]
        chunk_text = tokenizer.decode(chunk)
        chunks.append(chunk_text)

    return chunks


@task
def chunking_abstracts(df):
    tokenizer = AutoTokenizer.from_pretrained(cfg.EMBED_MODEL)
    articles_chunks = []

    for n, row in df.iterrows():
        text = "Title: " + row["title"] + " - Abstract:" + row["abstract"]
        text = clean_text(text)
        chunks = chunk_by_tokens(
            text=text,
            tokenizer=tokenizer,
            chunk_size=cfg.CHUNK_SIZE,
            overlap=cfg.CHUNK_OVERLAP,
        )
        for i, chunk in enumerate(chunks):
            articles_chunks.append(
                {
                    "id": n,
                    "title": row.title,
                    "categories": row.categories,
                    "abstract_chunk": chunk,
                    "id_chunk": i,
                    "authors": row.authors,
                    "url": row.url,
                    "published": row.published,
                    "year": row.year,
                }
            )
    df_final = pd.DataFrame(data=articles_chunks)
    return df_final


@task
def generate_embeddings(df):
    model = TextEmbedding(
        model_name=cfg.EMBED_MODEL,
        provider="torch",
        device="cuda" if cfg.GPU else "cpu",
    )
    embeds = []
    for _, art in df.iterrows():
        vector = list(model.embed(art["abstract_chunk"]))[0]
        if len(vector) > 0:
            embeds.append(vector)

    df["embed"] = embeds
    return df


def sanitize_payload(art):
    return {
        "id": int(art.id),
        "title": str(art.title or ""),
        "abstract_chunk": str(art.abstract_chunk or ""),
        "categories": str(art.categories or ""),
        "authors": str(art.authors or ""),
        "published": str(art.published or ""),
        "year": str(art.year or ""),
        "url": str(art.url or ""),
    }


@task
def init_qdrant_client():
    client = QdrantClient(url=cfg.QDRANT_URL, api_key=cfg.QDRANT_API_KEY, timeout=60.0)
    return client


@task
def init_collection(
    client=None, collection_name="articles-rag", embed_dim=384, metric="cosine"
):
    distances = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclidean": models.Distance.EUCLID,
        "manhattan": models.Distance.MANHATTAN,
    }

    try:
        client.delete_collection(collection_name=collection_name)
    except Exception as e:
        print(f"Error: {e}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embed_dim, distance=distances[metric], on_disk=False
        ),
    )
    return


@task
def create_vdb(
    data,
    collection_name="articles-rag",
    embed_dim=384,
    metric="cosine",
    batch_size=1000,
):
    client = init_qdrant_client()
    init_collection(
        client=client,
        collection_name=collection_name,
        embed_dim=embed_dim,
        metric=metric,
    )

    id_ = 0
    points = []

    for _, art in data.iterrows():
        vector = art["embed"]

        if vector is None or len(vector) != embed_dim:
            print(f"Skipping id {art.id} — invalid embedding: {vector}")
            continue

        point = models.PointStruct(id=id_, vector=vector, payload=sanitize_payload(art))

        points.append(point)
        id_ += 1

        if len(points) == batch_size:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            points = []

    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True)

    client.create_payload_index(
        collection_name=cfg.COLLECTION,
        field_name="categories",
        field_schema=models.PayloadSchemaType.TEXT,
    )

    return points


@flow(name="Articles RAG Pipeline", retries=1, retry_delay_seconds=300)
def articles_pipeline():
    categories = cfg.ARXIV_CATEGORIES.items()
    keywords = sum(cfg.TRENDING_KEYWORDS.values(), [])

    df = download_arxiv_articles(
        years=cfg.YEARS, categories=categories, keywords=keywords
    )
    df = clean_data(df)
    save_data(df=df, path=cfg.PATH_ARTICLES, filename="articles.parquet")

    df = chunking_abstracts(df)
    save_data(df=df, path=cfg.PATH_ARTICLES, filename="articles_chunks.parquet")

    df = generate_embeddings(df)
    save_data(df=df, path=cfg.PATH_ARTICLES, filename="articles_w_embeds.parquet")

    create_vdb(
        data=df,
        collection_name=cfg.COLLECTION,
        embed_dim=cfg.EMBEDDING_DIMENSIONALITY,
        metric=cfg.DISTANCE_METRIC,
        batch_size=cfg.BATCH_SIZE,
    )
    return


if __name__ == "__main__":
    articles_pipeline()
