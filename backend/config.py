from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Model configurations
MODEL_CONFIG = {
    "model_name": "sentence-transformers/all-mpnet-base-v2",
    "model_kwargs": {"device": "cpu"},
    "encode_kwargs": {"normalize_embeddings": True}
}

# API configurations
API_SETTINGS = {
    "host": "0.0.0.0",
    "port": 8000,
    "cors_origins": ["*"]
}