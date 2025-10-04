📚 AI Knowledge Base Q&A — RAG + PyTorch Reranker + FastAPI + Docker

A lightweight Retrieval-Augmented Generation (RAG) system designed for CPU-only environments.

It provides a FastAPI service that answers natural language questions from your organization’s knowledge base using:
Embeddings + FAISS for fast retrieval

PyTorch Cross-Encoder Reranker (BERT-base) for improved relevance

FLAN-T5-small (local) for grounded answer generation

Docker container for reproducible, CPU-friendly deployment

🚀 Features

Endpoints:
    GET /health → status check
    
    POST /query → ask a question, get {answer, sources}
    
    POST /train → fine-tune reranker on synthetic or labeled pairs (runs in background)
    
    
Tech choices:

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Vector DB: FAISS (local index)

Reranker: bert-base-uncased fine-tuned on synthetic FAQ pairs

Generator: google/flan-t5-small (fast on CPU)

    Persistence:
    data/ → raw KB + training pairs

    models/ → saved FAISS index + reranker weights
    
    Deployment: CPU-only Docker image with mounted volumes for persistence

    
    📂 Project Layout
ai-kb-rag-pytorch/
│── app/
│   ├── main.py          # FastAPI routes
│   ├── rag_pipeline.py  # embed → retrieve → rerank → generate
│   ├── torch_reranker.py# PyTorch model + train/infer
│   ├── generator.py     # flan-t5 inference (HF, CPU)
│   ├── data_utils.py    # chunking, FAISS index, synthetic pairs
│   └── config.py
│── data/
│   ├── kb.txt           # Knowledge base (FAQs/policies)
│   └── pairs.jsonl      # (auto-generated) training pairs
│── models/
│   └── reranker/        # fine-tuned PyTorch weights
│── requirements.txt
│── Dockerfile
│── README.md


    ⚡ Quick Start

    1. Clone & build
git clone https://github.com/<your-username>/ai-kb-rag-pytorch.git
cd ai-kb-rag-pytorch

docker build -t ai-kb-rag:cpu .

    2. Run container
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  ai-kb-rag:cpu

    3. Check health
curl http://127.0.0.1:8000/health
# {"status":"ok"}

    🔍 Example Queries
# Ask a question
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How to request a refund?"}' | jq

    Response:
{
  "answer": "Email support@acme.com with your order ID and reason. Refunds are processed in 5–7 business days.",
  "sources": [
    {"chunk_id": 0, "score": 1.0, "snippet": "Email support@acme.com with your order ID and reason..."}
  ]
}

    Train the reranker
curl -X POST http://127.0.0.1:8000/train
# {"status":"training_started"}
🛠️ Roadmap
 Redis cache for repeated queries
 More KB formats (PDF ingestion)
 TorchScript export for reranker (faster CPU inference)


