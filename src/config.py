from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # remonte 1 ou 2 niveaux selon ton projet

class Config:
    DATA_DIR = ROOT / "data"
    PERSIST_DIRECTORY = DATA_DIR / "chroma_db"
    PERSIST_DATABASE = DATA_DIR / "database.db"
    SAVE_DOCUMENTS_DIR = DATA_DIR / "documents"
    LOG_FILE = DATA_DIR / "logs" / "rag_logs.jsonl"
    EMBEDDING_MODEL = "text-embedding-3-small"
    VECTOR_STORE_COLLECTION = "legal_documents"
