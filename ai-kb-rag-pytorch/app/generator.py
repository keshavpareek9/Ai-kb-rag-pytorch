from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .config import GEN_MODEL, MAX_GEN_TOKENS, DEVICE

class AnswerGenerator:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(GEN_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
        self.model.to(DEVICE)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, question: str, context: str) -> str:
        prompt = (
            "You must answer ONLY from the context.\n"
            "If the answer is present as an FAQ, return the text that follows 'A:' for the matching 'Q:'.\n"
            "If unsure, reply exactly: \"I don't know from the KB.\"\n"
            "No headings, no extra sentences.\n\n"
            f"Question: {question}\n"
            f"Context:\n{context}\n"
            "Answer:"
        )
        inp = self.tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        out = self.model.generate(**inp, max_new_tokens=MAX_GEN_TOKENS)
        return self.tok.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
