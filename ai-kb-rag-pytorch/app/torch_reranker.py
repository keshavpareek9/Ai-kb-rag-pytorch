from pathlib import Path
from typing import List, Tuple
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
)
from .config import RERANKER_BASE, RERANKER_DIR, DEVICE, TRAIN_EPOCHS, TRAIN_LR, TRAIN_BATCH, PAIRS_FILE

class PairDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len=256):
        self.items = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        enc = self.tok(
            ex["query"], ex["chunk"],
            truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        enc = {k: v.squeeze(0) for k,v in enc.items()}
        label = torch.tensor(ex["label"], dtype=torch.float)
        return {**enc, "labels": label}

def train_reranker(pairs_path=PAIRS_FILE, output_dir=RERANKER_DIR):
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_BASE, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_BASE, num_labels=1
    )
    ds = PairDataset(pairs_path, tokenizer)

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=TRAIN_LR,
        per_device_train_batch_size=TRAIN_BATCH,
        num_train_epochs=TRAIN_EPOCHS,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=1,
        report_to=[]
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return str(output_dir)

class CrossEncoder:
    def __init__(self, model_dir=RERANKER_DIR):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(DEVICE)
        self.model.eval()

    @torch.inference_mode()
    def score(self, query: str, passages: List[str]) -> List[float]:
        batch = self.tokenizer(
            [query]*len(passages), passages,
            truncation=True, padding=True, max_length=256, return_tensors="pt"
        )
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = self.model(**batch).logits.squeeze(-1)
        return logits.cpu().tolist()

    def export_torchscript(self, out_path: Path):
        # TorchScript for the transformer encoder head.
        # Note: tokenization still runs in Python; this speeds up the model forward on CPU.
        example = {k: torch.ones(1, 256, dtype=torch.long) for k in ["input_ids","attention_mask","token_type_ids"]}
        traced = torch.jit.trace(self.model, example)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(out_path))
        return str(out_path)
