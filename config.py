import os
from dotenv import load_dotenv
import torch

load_dotenv()

PATH_ARTICLES = "./data/"

YEARS = 7

ARXIV_PAGE_SIZE = 2000
ARXIV_DELAY = 10.0
ARXIV_RETRIES = 5
ARXIV_MAX_RESULTS = 15000

EMBED_MODEL = "BAAI/bge-small-en"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

EMBEDDING_DIMENSIONALITY = 384
BATCH_SIZE = 1000

LOCAL_DEPLOYMENT = False
GPU = torch.cuda.is_available()
QDRANT_URL = (
    "https://6740984a-35a5-4d5c-bc6e-b7288f2991c7.us-east4-0.gcp.cloud.qdrant.io:6333"
    if not LOCAL_DEPLOYMENT
    else "http://qdrant:6333"
)
QDRANT_API_KEY = os.getenv("QDRANT_CLOUD_API_KEY") if not LOCAL_DEPLOYMENT else None
DISTANCE_METRIC = "cosine"
COLLECTION = "articles-rag-cos"

GEMINI_MODEL = "gemini-2.5-flash-lite"

os.environ["PREFECT_API_URL"] = (
    "http://127.0.0.1:4200/api" if not LOCAL_DEPLOYMENT else "http://prefect:4200/api"
)

ARXIV_CATEGORIES = {
    "cs.LG": "Machine Learning",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CL": "Computation and Language",
    "cs.AI": "Artificial Intelligence",
    "stat.ML": "Statistics",
    "eess.IV": "Electrical Engineering and Systems Science",
    "cs.RO": "Robotics",
    "cs.NE": "Neural and Evolutionary Computing",
}

TRENDING_KEYWORDS = {
    "model_architecture": [
        "transformer",
        "BERT",
        "GPT",
        "LLaMA",
        "vision transformer",
        "ViT",
        "GNN",
        "CNN",
        "RNN",
        "autoencoder",
        "variational autoencoder",
        "attention mechanism",
        "GAN",
    ],
    "techniques": [
        "self-supervised learning",
        "unsupervised learning",
        "reinforcement learning",
        "contrastive learning",
        "few-shot learning",
        "zero-shot learning",
        "meta-learning",
        "transfer learning",
        "foundation model",
        "multimodal learning",
    ],
    "applications": [
        "computer vision",
        "image classification",
        "object detection",
        "segmentation",
        "NLP",
        "speech recognition",
        "ASR",
        "robotics",
        "autonomous systems",
        "bioinformatics",
        "recommendation system",
    ],
    "content_generation": [
        "text generation",
        "image generation",
        "diffusion model",
        "LLM",
        "retrieval augmented generation",
    ],
}
