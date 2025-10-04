from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RERANKER_DIR = MODELS_DIR / "reranker"
FAISS_DIR = MODELS_DIR / "faiss"
FAISS_DIR.mkdir(parents=True, exist_ok=True)
RERANKER_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

KB_FILE = DATA_DIR / "kb.txt"
PAIRS_FILE = DATA_DIR / "pairs.jsonl"
INDEX_FILE = FAISS_DIR / "index.faiss"
CHUNKS_FILE = FAISS_DIR / "chunks.json"

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_BASE = "bert-base-uncased"
GEN_MODEL = "google/flan-t5-small"

# Retrieval / generation
TOP_K = 4
TOP_K_RERANK = 2
MAX_INPUT_TOKENS = 512
MAX_GEN_TOKENS = 256

# Training
TRAIN_EPOCHS = 1
TRAIN_LR = 2e-5
TRAIN_BATCH = 16

# Optional Redis
USE_REDIS = False  # set True if you run Redis
REDIS_URL = "redis://localhost:6379/0"

DEVICE = "cpu"  # keep CPU-only
