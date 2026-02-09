#!/usr/bin/env python3
"""
LLM explanation generation with optional retrieval (few-shot / RAG) and metrics logging.

Usage example:
python run_generate.py \
  --hf-token ... \
  --dataset wiki \
  --test-size 200 \
  --samples 3 \
  --function sbert \
  --prompt rag \
  --model ... \
  --top-p 0.95 \
  --top-k 50 \
  --temp 0.3 \
  --max-tokens 256 \
  --penalty 1.0 \
  --output outputs/wiki_rag_sbert.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Your utils
from utils.a5evaluates import compute_metrics


# -----------------------------
# Config / CLI
# -----------------------------
@dataclass
class RunConfig:
    hf_token: str
    dataset: str            # wiki | exa
    test_size: int
    samples: int
    function: str           # few | sbert | bm25 | tfidf
    prompt: str             # zero | few | rag
    model: str
    top_p: float
    top_k: int
    temp: float
    max_tokens: int
    penalty: float
    output: str
    batch_size: int
    metrics_log: str


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Generate explanations with optional retrieval + log metrics.")
    parser.add_argument("--hf-token", type=str, required=True, help="HuggingFace token")
    parser.add_argument("--dataset", type=str, required=True, choices=["wiki", "exa"], help="Dataset: wiki or exa")
    parser.add_argument("--test-size", type=int, required=True, help="Number of test samples")
    parser.add_argument("--samples", type=int, default=3, help="k for few-shot / RAG retrieval")
    parser.add_argument("--function", type=str, required=True, choices=["few", "sbert", "bm25", "tfidf"],
                        help="Retrieval function: few, sbert, bm25, tfidf")
    parser.add_argument("--prompt", type=str, required=True, choices=["zero", "few", "rag"],
                        help="Prompt type: zero, few, rag")
    parser.add_argument("--model", type=str, required=True, help="HF model name/path")
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--temp", type=float, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--penalty", type=float, required=True)
    parser.add_argument("--output", type=str, required=True, help="CSV path for generation outputs")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for LLM generation")
    parser.add_argument("--metrics-log", type=str, default="experiment_results.csv", help="Master CSV for metrics")

    a = parser.parse_args()
    return RunConfig(
        hf_token=a.hf_token,
        dataset=a.dataset,
        test_size=a.test_size,
        samples=a.samples,
        function=a.function,
        prompt=a.prompt,
        model=a.model,
        top_p=a.top_p,
        top_k=a.top_k,
        temp=a.temp,
        max_tokens=a.max_tokens,
        penalty=a.penalty,
        output=a.output,
        batch_size=a.batch_size,
        metrics_log=a.metrics_log,
    )


# -----------------------------
# Logging
# -----------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    # Quiet transformers
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()


# -----------------------------
# Dataset
# -----------------------------
def get_hf_dataset_name(dataset_key: str) -> str:
    if dataset_key == "wiki":
        return "data/wikisa1/train.csv", "data/wikisa1/test.csv"
    if dataset_key == "exa":
        return "data/exarank1/train.csv", "data/exarank1/test.csv"
    raise ValueError(f"Unknown dataset: {dataset_key}")


def load_splits(ds_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(ds_name[0])
    test = pd.read_csv(ds_name[1])

    return train, test


def add_docs_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["docs"] = df["query"] + " [SEP] " + df["doc"]
    return df


# -----------------------------
# Retrieval
# -----------------------------
def retrieval_few(train_df: pd.DataFrame, test_df: pd.DataFrame, test_size: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train = train_df.to_dict(orient="records")
    test = test_df.sample(n=test_size, random_state=42).to_dict(orient="records")
    return train, test


def retrieval_sbert(train_df: pd.DataFrame, test_df: pd.DataFrame, test_size: int, k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    import faiss
    from sentence_transformers import SentenceTransformer

    train = train_df.to_dict(orient="records")
    docs_train = [r["docs"] for r in train]

    model = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base").to("cuda")

    logging.info("Encoding train embeddings...")
    train_emb = model.encode(docs_train, batch_size=256, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(train_emb)

    logging.info("Building FAISS index...")
    index = faiss.IndexFlatIP(train_emb.shape[1])
    index.add(train_emb)

    test_df = test_df.head(test_size)
    test = test_df.to_dict(orient="records")
    docs_test = [r["docs"] for r in test]

    logging.info("Encoding test embeddings...")
    test_emb = model.encode(docs_test, batch_size=256, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(test_emb)

    logging.info("Retrieving top-k via FAISS...")
    _, indices = index.search(test_emb, k)

    for i, row in enumerate(test):
        row["retrieved_indices"] = indices[i].tolist()

    return train, test


def retrieval_bm25(ds_key: str, ds_name: str, test_df: pd.DataFrame, test_size: int, k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cache_file = "./utils/bm_25/bm25s_topk_wiki.pkl" if ds_key == "wiki" else "./utils/bm_25/bm25s_topk_exa.pkl"
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"{cache_file} not found. Run bm25s_builder.py first.")

    logging.info("Loading BM25 cache: %s", cache_file)
    with open(cache_file, "rb") as f:
        cache = pickle.load(f)

    if cache.get("ds_name") != ds_name:
        raise ValueError(f"Cache dataset ({cache.get('ds_name')}) != requested dataset ({ds_name})")

    train = cache["train"]
    all_indices = cache["all_indices"]

    test_df = test_df.head(test_size)
    test = test_df.to_dict(orient="records")

    if len(all_indices) < len(test):
        raise ValueError(f"Not enough cached indices for test_size={test_size}")

    for i, row in enumerate(test):
        row["retrieved_indices"] = all_indices[i][:k]

    return train, test


def retrieval_tfidf(train_df: pd.DataFrame, test_df: pd.DataFrame, test_size: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Any, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(train_df["docs"].tolist())

    train = train_df.to_dict(orient="records")
    test = test_df.head(test_size).to_dict(orient="records")
    return train, test, vectorizer, tfidf_matrix


# -----------------------------
# Prompt building
# -----------------------------
def build_prompt(
    cfg: RunConfig,
    row: Dict[str, Any],
    train: List[Dict[str, Any]],
    vectorizer: Any = None,
    tfidf_matrix: Any = None,
) -> str:
    q, d = row["query"], row["doc"]

    # Import only when needed to keep dependencies tidy
    from utils import a5prompts as P

    if cfg.dataset == "wiki":
        if cfg.prompt == "zero":
            return P.wiki_zero_prompt(q, d)

        if cfg.prompt == "few":
            if cfg.function != "few":
                raise ValueError("Prompt=few requires function=few (random few-shot).")
            res = P.rank_random(train, topk=cfg.samples)
            return P.wiki_few_prompt(q, d, res)

        if cfg.prompt == "rag":
            if cfg.function in {"sbert", "bm25"}:
                res = [train[idx] for idx in row["retrieved_indices"]]
                return P.wiki_rag_prompt(q, d, res)
            if cfg.function == "tfidf":
                res = P.rank_tfidf(d, vectorizer, tfidf_matrix, train, topk=cfg.samples)
                return P.wiki_rag_prompt(q, d, res)

    if cfg.dataset == "exa":
        if cfg.prompt == "zero":
            return P.exa_zero_prompt(q, d)

        if cfg.prompt == "few":
            # Your original script uses exa_rag_prompt for few-shot
            res = P.rank_random(train, topk=cfg.samples)
            return P.exa_rag_prompt(q, d, res)

        if cfg.prompt == "rag":
            if cfg.function in {"sbert", "bm25"}:
                res = [train[idx] for idx in row["retrieved_indices"]]
                return P.exa_rag_prompt(q, d, res)
            if cfg.function == "tfidf":
                res = P.rank_tfidf(d, vectorizer, tfidf_matrix, train, topk=cfg.samples)
                return P.exa_rag_prompt(q, d, res)

    raise ValueError(f"Unsupported combination: dataset={cfg.dataset}, prompt={cfg.prompt}, function={cfg.function}")


def extract_explanation(dataset_key: str, generated_text: str) -> str:
    if dataset_key == "wiki":
        matches = re.findall(r"aspect:\s*(.*)", generated_text, flags=re.IGNORECASE)
    else:
        matches = re.findall(r"explanation:\s*(.*)", generated_text, flags=re.IGNORECASE)

    return matches[-1].strip() if matches else "[NO_MATCH]"


# -----------------------------
# Model loading / generation
# -----------------------------
def load_llm(cfg: RunConfig) -> Tuple[Any, Any, torch.device]:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = cfg.hf_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    llm = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # important for batch generation

    return llm, tokenizer, device


@torch.no_grad()
def generate_batches(
    llm: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: List[str],
    cfg: RunConfig,
) -> List[str]:
    outputs_text: List[str] = []

    do_sample = cfg.temp > 0.0

    for start in tqdm(range(0, len(prompts), cfg.batch_size), desc="Batch generation"):
        batch_prompts = prompts[start:start + cfg.batch_size]

        inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        out_ids = llm.generate(
            **inputs,
            max_new_tokens=cfg.max_tokens,
            temperature=cfg.temp,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.penalty,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
        )

        batch_texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        outputs_text.extend(batch_texts)

    return outputs_text


# -----------------------------
# Saving + logging metrics
# -----------------------------
def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def save_outputs(path: str, refs: List[str], full_texts: List[str], extracted: List[str]) -> None:
    ensure_parent_dir(path)
    df = pd.DataFrame({"ref": refs, "llm": full_texts, "exp": extracted})
    df.to_csv(path, index=False)
    logging.info("Saved generations to: %s", path)


def append_metrics_log(path: str, cfg: RunConfig, metrics: Dict[str, Any]) -> None:
    row = asdict(cfg)
    row.update(metrics)

    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header)
    logging.info("Appended metrics row to: %s", path)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    setup_logging()
    cfg = parse_args()

    logging.info("Args: %s", cfg)

    ds_name = get_hf_dataset_name(cfg.dataset)

    train_df, _, test_df = load_splits(ds_name)
    train_df = add_docs_column(train_df)
    test_df = add_docs_column(test_df)

    # Retrieval stage
    vectorizer, tfidf_matrix = None, None

    if cfg.function == "few":
        train, test = retrieval_few(train_df, test_df, cfg.test_size)

    elif cfg.function == "sbert":
        train, test = retrieval_sbert(train_df, test_df, cfg.test_size, cfg.samples)

    elif cfg.function == "bm25":
        train, test = retrieval_bm25(cfg.dataset, ds_name, test_df, cfg.test_size, cfg.samples)

    elif cfg.function == "tfidf":
        train, test, vectorizer, tfidf_matrix = retrieval_tfidf(train_df, test_df, cfg.test_size)

    else:
        raise ValueError(f"Unknown retrieval function: {cfg.function}")

    # Build prompts + refs
    prompts: List[str] = []
    refs: List[str] = []

    logging.info("Building prompts (dataset=%s, prompt=%s, function=%s)...", cfg.dataset, cfg.prompt, cfg.function)
    for row in tqdm(test, desc="Building prompts"):
        prompts.append(build_prompt(cfg, row, train, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix))
        refs.append(row["explanation"])

    # Load model and generate
    llm, tokenizer, device = load_llm(cfg)

    logging.info("Generating outputs...")
    full_generations = generate_batches(llm, tokenizer, device, prompts, cfg)

    extracted = [extract_explanation(cfg.dataset, txt) for txt in full_generations]

    # Save outputs
    save_outputs(cfg.output, refs, full_generations, extracted)

    # Metrics (compute once)
    metrics = compute_metrics(extracted, refs)
    logging.info("Metrics: %s", metrics)

    append_metrics_log(cfg.metrics_log, cfg, metrics)

    logging.info("Done!")


if __name__ == "__main__":
    main()
