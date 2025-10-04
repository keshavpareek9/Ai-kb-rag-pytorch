import json, re
from typing import Dict, List, Tuple
from .config import TOP_K, TOP_K_RERANK
from .data_utils import build_or_load_faiss, embed_queries
from .torch_reranker import CrossEncoder
from .generator import AnswerGenerator

def _clean(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

class RagPipeline:
    def __init__(self):
        self.index, self.chunks = build_or_load_faiss()
        self.reranker = self._try_load_reranker()
        self.generator = AnswerGenerator()
        self.cache = None  # plug Redis here if you enable it

    def _try_load_reranker(self):
        try:
            return CrossEncoder()
        except Exception:
            return None

    def _faiss_topk(self, q: str, k: int) -> List[Tuple[int, float]]:
        # clamp k to available chunks
        k = max(0, min(k, len(self.chunks)))
        if k == 0:
            return []
        vec = embed_queries([q])
        D, I = self.index.search(vec, k)
        # filter out FAISS's -1 placeholders
        pairs = [(int(i), float(d)) for i, d in zip(I[0], D[0]) if int(i) >= 0]
        return pairs

    def _extract_snippet(self, passage: str, question: str) -> str:
        """Prefer the answer that follows the best matching 'Q:'; fallback to first sentence."""
        q_lower = question.lower()

        # 1) Line-based: find a "Q:" line that best overlaps the question, then grab the next "A:" line
        lines = [l.strip() for l in passage.splitlines() if l.strip()]
        best_q_idx, best_overlap = None, 0
        for i, line in enumerate(lines):
            if line.lower().startswith("q:"):
                qw = set(re.findall(r"\w+", line.lower()))
                uw = set(re.findall(r"\w+", q_lower))
                overlap = len(qw & uw)
                if overlap > best_overlap:
                    best_overlap, best_q_idx = overlap, i
        if best_q_idx is not None:
            for j in range(best_q_idx + 1, min(best_q_idx + 6, len(lines))):
                if lines[j].lower().startswith("a:"):
                    return _clean(lines[j][2:].strip())

        # 2) Paragraph-based: capture "...Q: ... A: <answer> ..." in the same blob
        m = re.search(r"q:\s*.*?a:\s*(.+?)(?:\n|$)", passage, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return _clean(m.group(1).strip())

        # 3) Fallback: first sentence
        first = re.split(r'(?<=[.!?])\s+', passage.strip())
        return _clean(first[0]) if first else _clean(passage[:200])

    def query(self, question: str) -> Dict:
        hits = self._faiss_topk(question, TOP_K)
        candidates = [(idx, self.chunks[idx]) for idx, _ in hits]
        if not candidates:
            return {"answer": "I don't know from the KB.", "sources": []}

        # Rerank if trained, otherwise just use FAISS order
        if self.reranker is not None and candidates:
            idxs, passages = zip(*candidates)
            scores = self.reranker.score(question, list(passages))
            reranked = sorted(zip(idxs, passages, scores), key=lambda x: x[2], reverse=True)
            chosen = reranked[:min(TOP_K_RERANK, len(reranked))]
        else:
            chosen = [(i, p, 0.0) for i, p in candidates[:min(TOP_K_RERANK, len(candidates))]]

        # Deterministic FAQ extraction from the top passage (prefer this if it looks like a proper answer)
        top_id, top_passage, top_score = chosen[0]
        faq_ans = self._extract_snippet(top_passage, question)
        if faq_ans and not faq_ans.lower().startswith("q:") and len(faq_ans.split()) > 3:
            return {
                "answer": faq_ans,
                "sources": [
                    {"chunk_id": int(i), "score": float(s), "snippet": self._extract_snippet(p, question)}
                    for i, p, s in chosen
                ]
            }

        # Otherwise, build context for the generator
        context = "\n\n".join([p for _, p, _ in chosen])
        answer = self.generator.generate(question, context)

        return {
            "answer": answer if answer else faq_ans,
            "sources": [
                {"chunk_id": int(i), "score": float(s), "snippet": self._extract_snippet(p, question)}
                for i, p, s in chosen
            ]
        }

    def refresh_indices(self):
        self.index, self.chunks = build_or_load_faiss()
