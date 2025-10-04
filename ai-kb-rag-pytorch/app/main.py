from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from .rag_pipeline import RagPipeline
from .data_utils import ensure_pairs, build_or_load_faiss
from .torch_reranker import train_reranker
from .config import PAIRS_FILE

app = FastAPI(title="AI KB RAG + PyTorch Reranker (CPU)")

# initialize pipeline at startup
pipeline = RagPipeline()

class QueryIn(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(q: QueryIn):
    return pipeline.query(q.question)

def _train_job():
    # regen FAISS/chunks (in case kb changed)
    index, chunks = build_or_load_faiss()
    # synth pairs from chunks if not present
    ensure_pairs(chunks, out_file=PAIRS_FILE)
    # train
    train_reranker(PAIRS_FILE)
    # hot-reload reranker
    pipeline.__init__()

@app.post("/train")
def train(background_tasks: BackgroundTasks):
    background_tasks.add_task(_train_job)
    return {"status": "training_started"}
