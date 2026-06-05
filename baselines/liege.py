import os
import copy
import math
import random
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import datasets
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,                      # type: ignore
    AutoModel,                          # type: ignore
    BartForConditionalGeneration,       # type: ignore
    get_linear_schedule_with_warmup,    # type: ignore
    logging,                            # type: ignore
)

logging.set_verbosity_error()

from utils.a5evaluates import compute_metrics


# =========================
# Arguments
# =========================
parser = argparse.ArgumentParser(description="LiEGe baseline from scratch with hyperparameter tuning")
parser.add_argument('--hf-dataset', type=str, required=True, help="HF dataset name, e.g. ariflaksito/wikicn1")
parser.add_argument('--test-size', type=int, required=True, help="Test sample size")
parser.add_argument('--target-len', type=int, default=32, help="Max generation length")
parser.add_argument('--lr', type=float, default=5e-5, help="Base learning rate")
parser.add_argument('--output-dir', type=str, required=True, help="Directory to save outputs")
parser.add_argument('--seed', type=int, default=42)

# tuning search space
parser.add_argument('--batch-sizes', type=int, nargs='+', default=[2, 4], help="SERP batch sizes to try")
parser.add_argument('--epochs-list', type=int, nargs='+', default=[3, 5], help="Epoch values to try")
parser.add_argument('--dropouts', type=float, nargs='+', default=[0.1], help="Dropout values to try")
parser.add_argument('--warmup-steps-list', type=int, nargs='+', default=[0, 100], help="Warmup steps to try")

# early stopping
parser.add_argument('--patience', type=int, default=2, help="Early stopping patience")
parser.add_argument('--min-delta', type=float, default=1e-4, help="Minimum val loss improvement")

# model params
parser.add_argument('--backbone', type=str, default='bart', choices=['bart', 'bert'], help="LiEGe starting backbone")
parser.add_argument('--bart-name', type=str, default='facebook/bart-base')
parser.add_argument('--bert-name', type=str, default='bert-base-uncased')
parser.add_argument('--num-local-layers', type=int, default=12)
parser.add_argument('--num-global-layers', type=int, default=2)
parser.add_argument('--num-decoder-layers', type=int, default=12)
parser.add_argument('--num-global-heads', type=int, default=8)
parser.add_argument('--max-len', type=int, default=512)
parser.add_argument('--max-docs', type=int, default=2)
parser.add_argument('--pooling', type=str, default='cls', choices=['cls', 'mean'])
parser.add_argument('--use-joint-target', action='store_true', help="Train both decoder slots with the same target: explanation1 and explanation2")
parser.add_argument('--use-ordinal', action='store_true', help="Add ordinal rank embedding to document tokens")
parser.add_argument('--gradient-clip', type=float, default=1.0)

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

print("Arguments:")
print(vars(args), flush=True)


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)


# =========================
# Helpers
# =========================
def safe_get(row: Dict[str, Any], candidates: List[str], default: str = "") -> str:
    for c in candidates:
        if c in row and row[c] is not None:
            return str(row[c])
    return default


def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)


class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, current_loss: float) -> bool:
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# =========================
# Dataset preparation
# =========================
@dataclass
class DocItem:
    query: str
    doc: str
    explanation: str
    rank: int


class LiEGESerpDataset(Dataset):
    """
      input  = query + [SEP] + doc1 + [SEP] + doc2
      target = explanation1 + ' and ' + explanation2
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_len: int = 512,
        max_target_len: int = 32,
        max_docs: int = 2,
        use_joint_target: bool = False,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_target_len = max_target_len
        self.max_docs = max_docs
        self.pad_token_id = tokenizer.pad_token_id
        self.use_joint_target = use_joint_target

        required_cols = ["query", "doc1", "doc2", "explanation1", "explanation2"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

    def __len__(self):
        return len(self.df)

    def _encode_source(self, query: str, doc1: str, doc2: str):
        # explicit concatenation using tokenizer special tokens
        text = f"{query} {self.tokenizer.sep_token} {doc1} {self.tokenizer.sep_token} {doc2}"
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors=None,
        )
        return enc["input_ids"], enc["attention_mask"]

    def _encode_target(self, text: str):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_target_len,
            return_tensors=None,
        )
        return enc["input_ids"], enc["attention_mask"]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        query = str(row["query"])
        doc1 = str(row["doc1"])
        doc2 = str(row["doc2"])
        exp1 = str(row["explanation1"])
        exp2 = str(row["explanation2"])

        src_ids, src_mask = self._encode_source(query, doc1, doc2)

        if self.use_joint_target:
            # optional mode if you want one concatenated explanation per item
            joint_target = f"{exp1} and {exp2}"
            t1_ids, t1_mask = self._encode_target(joint_target)
            t2_ids, t2_mask = self._encode_target(joint_target)
        else:
            # default LiEGe-compatible mode: two decoded outputs
            t1_ids, t1_mask = self._encode_target(exp1)
            t2_ids, t2_mask = self._encode_target(exp2)

        return {
            "src_input_ids": torch.tensor([src_ids, src_ids], dtype=torch.long),
            "src_attention_mask": torch.tensor([src_mask, src_mask], dtype=torch.long),
            "tgt_input_ids": torch.tensor([t1_ids, t2_ids], dtype=torch.long),
            "tgt_attention_mask": torch.tensor([t1_mask, t2_mask], dtype=torch.long),
            "doc_mask": torch.tensor([1, 1], dtype=torch.bool),
            "doc_ranks": torch.tensor([1, 2], dtype=torch.long),
            "joint_reference": f"{exp1} and {exp2}",
        }


# =========================
# LiEGe modules
# =========================
class MultiHeadPooling(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.proj = nn.Linear(hidden_size, num_heads)
        self.out = nn.Linear(hidden_size * num_heads, hidden_size)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, K, L, H]
        # attn_mask: [B, K, L] -> 1 valid, 0 pad
        scores = self.proj(x)                                  # [B, K, L, heads]
        scores = scores.masked_fill(attn_mask.unsqueeze(-1) == 0, -1e9)
        weights = torch.softmax(scores, dim=2)                 # over tokens
        pooled = torch.einsum('bklh,bklm->bkhm', weights, x)    # [B, K, heads, H]
        pooled = pooled.reshape(x.size(0), x.size(1), -1)      # [B, K, heads*H]
        return self.out(pooled)                                # [B, K, H]


class LocalEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # x: [B*K, L, H]
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.ffn(x)
        x = self.norm2(x + self.dropout(ff))
        return x


class GlobalEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ff_dim: int, dropout: float, pooling: str = 'cls'):
        super().__init__()
        self.pooling = pooling
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm_docs_1 = nn.LayerNorm(hidden_size)
        self.norm_docs_2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
        )
        self.mhp = MultiHeadPooling(hidden_size, num_heads=num_heads)

    def pool_docs(self, token_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        # token_states: [B, K, L, H], token_mask: [B, K, L]
        if self.pooling == 'cls':
            return token_states[:, :, 0, :]
        return self.mhp(token_states, token_mask)

    def forward(
        self,
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # token_states: [B, K, L, H]
        # token_mask:   [B, K, L] (1 valid)
        # doc_mask:     [B, K]    (1 real doc)
        doc_states = self.pool_docs(token_states, token_mask)  # [B, K, H]

        key_padding_mask = ~doc_mask                           # MultiheadAttention expects True for pad
        attn_out, _ = self.self_attn(doc_states, doc_states, doc_states, key_padding_mask=key_padding_mask)
        doc_states = self.norm_docs_1(doc_states + self.dropout(attn_out))
        ff = self.ffn(doc_states)
        doc_states = self.norm_docs_2(doc_states + self.dropout(ff))

        # broadcast & add
        token_states = token_states + doc_states.unsqueeze(2)
        token_states = token_states * token_mask.unsqueeze(-1)
        return token_states, doc_states


class LiEGEDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_doc_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_tok_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.norm4 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
        )

    def forward(
        self,
        tgt_states: torch.Tensor,
        doc_states: torch.Tensor,
        token_states: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor],
        doc_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # tgt_states: [N, T, H]
        # doc_states: [N, K, H]
        # token_states: [N, S, H]
        x, _ = self.self_attn(
            tgt_states, tgt_states, tgt_states,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt_states = self.norm1(tgt_states + self.dropout(x))

        x, _ = self.cross_doc_attn(
            tgt_states, doc_states, doc_states,
            key_padding_mask=doc_key_padding_mask,
        )
        tgt_states = self.norm2(tgt_states + self.dropout(x))

        x, _ = self.cross_tok_attn(
            tgt_states, token_states, token_states,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt_states = self.norm3(tgt_states + self.dropout(x))

        ff = self.ffn(tgt_states)
        tgt_states = self.norm4(tgt_states + self.dropout(ff))
        return tgt_states


class LiEGeFromScratch(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        tokenizer,
        backbone_type: str = 'bart',
        num_local_layers: int = 12,
        num_global_layers: int = 2,
        num_decoder_layers: int = 12,
        num_global_heads: int = 8,
        dropout: float = 0.1,
        max_docs: int = 10,
        pooling: str = 'cls',
        use_ordinal: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone_type = backbone_type
        self.max_docs = max_docs
        self.pooling = pooling
        self.use_ordinal = use_ordinal
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

        if backbone_type == 'bart':
            bart = BartForConditionalGeneration.from_pretrained(backbone_name)
            self.shared = bart.model.shared
            self.hidden_size = bart.config.d_model
            self.vocab_size = bart.config.vocab_size
            self.encoder_num_heads = bart.config.encoder_attention_heads
            self.decoder_num_heads = bart.config.decoder_attention_heads
            self.ff_dim = bart.config.encoder_ffn_dim
            self.embed_scale = math.sqrt(self.hidden_size)
            self.pos_encoder = bart.model.encoder.embed_positions
            self.pos_decoder = bart.model.decoder.embed_positions
        else:
            bert = AutoModel.from_pretrained(backbone_name)
            self.shared = bert.embeddings.word_embeddings
            self.hidden_size = bert.config.hidden_size
            self.vocab_size = bert.config.vocab_size
            self.encoder_num_heads = bert.config.num_attention_heads
            self.decoder_num_heads = bert.config.num_attention_heads
            self.ff_dim = bert.config.intermediate_size
            self.embed_scale = 1.0
            self.pos_encoder = nn.Embedding(2048, self.hidden_size)
            self.pos_decoder = nn.Embedding(2048, self.hidden_size)

        self.emb_dropout = nn.Dropout(dropout)
        self.ordinal_embedding = nn.Embedding(max_docs + 2, self.hidden_size) if use_ordinal else None

        self.local_layers = nn.ModuleList([
            LocalEncoderLayer(
                hidden_size=self.hidden_size,
                num_heads=self.encoder_num_heads,
                ff_dim=self.ff_dim,
                dropout=dropout,
            ) for _ in range(num_local_layers)
        ])

        self.global_layers = nn.ModuleList([
            GlobalEncoderLayer(
                hidden_size=self.hidden_size,
                num_heads=num_global_heads,
                ff_dim=self.ff_dim,
                dropout=dropout,
                pooling=pooling,
            ) for _ in range(num_global_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            LiEGEDecoderLayer(
                hidden_size=self.hidden_size,
                num_heads=self.decoder_num_heads,
                ff_dim=self.ff_dim,
                dropout=dropout,
            ) for _ in range(num_decoder_layers)
        ])

        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.lm_head.weight = self.shared.weight
        self.final_ln = nn.LayerNorm(self.hidden_size)

    def _embed_encoder_inputs(self, input_ids: torch.Tensor, doc_ranks: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_ids: [B, K, L]
        B, K, L = input_ids.shape
        device = input_ids.device

        tok = self.shared(input_ids) * self.embed_scale
        pos_ids = torch.arange(L, device=device).unsqueeze(0).unsqueeze(0).expand(B, K, L)

        if self.backbone_type == 'bart':
            # BART embed_positions expects [B, L] shaped ids-like tensor
            pos = self.pos_encoder(torch.zeros(B * K, L, dtype=torch.long, device=device)).view(B, K, L, -1)
        else:
            pos = self.pos_encoder(pos_ids)

        x = tok + pos

        if self.ordinal_embedding is not None and doc_ranks is not None:
            clipped_ranks = doc_ranks.clamp(min=0, max=self.max_docs + 1)
            ord_emb = self.ordinal_embedding(clipped_ranks).unsqueeze(2)   # [B, K, 1, H]
            x = x + ord_emb

        return self.emb_dropout(x)

    def _embed_decoder_inputs(self, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        # decoder_input_ids: [N, T]
        N, T = decoder_input_ids.shape
        device = decoder_input_ids.device
        tok = self.shared(decoder_input_ids) * self.embed_scale

        if self.backbone_type == 'bart':
            pos = self.pos_decoder(torch.zeros(N, T, dtype=torch.long, device=device))
        else:
            pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(N, T)
            pos = self.pos_decoder(pos_ids)

        return self.emb_dropout(tok + pos)

    def encode(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        doc_ranks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # src_input_ids: [B, K, L]
        B, K, L = src_input_ids.shape
        x = self._embed_encoder_inputs(src_input_ids, doc_ranks=doc_ranks)  # [B, K, L, H]

        flat_x = x.view(B * K, L, self.hidden_size)
        flat_pad = (src_attention_mask.view(B * K, L) == 0)

        for layer in self.local_layers:
            flat_x = layer(flat_x, key_padding_mask=flat_pad)

        token_states = flat_x.view(B, K, L, self.hidden_size)
        token_mask = src_attention_mask.bool()

        final_doc_states = None
        for layer in self.global_layers:
            token_states, final_doc_states = layer(token_states, token_mask, doc_mask)

        if final_doc_states is None:
            # if num_global_layers == 0, create document representations from final token states
            if self.pooling == 'cls':
                final_doc_states = token_states[:, :, 0, :]
            else:
                denom = token_mask.sum(dim=2, keepdim=True).clamp(min=1)
                final_doc_states = (token_states * token_mask.unsqueeze(-1)).sum(dim=2) / denom

        # flatten docs for parallel decoding: [B*K, ...]
        enc_tok = token_states.view(B * K, L, self.hidden_size)
        enc_doc = final_doc_states.unsqueeze(1).expand(B, K, K, self.hidden_size).contiguous().view(B * K, K, self.hidden_size)
        enc_doc_mask = doc_mask.unsqueeze(1).expand(B, K, K).contiguous().view(B * K, K)
        return enc_tok, enc_doc, enc_doc_mask

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        enc_tok: torch.Tensor,
        enc_doc: torch.Tensor,
        enc_tok_attention_mask: torch.Tensor,
        enc_doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        # decoder_input_ids: [N, T]
        N, T = decoder_input_ids.shape
        x = self._embed_decoder_inputs(decoder_input_ids)
        tgt_mask = generate_square_subsequent_mask(T, decoder_input_ids.device)
        tgt_pad_mask = (decoder_input_ids == self.pad_token_id)
        memory_pad_mask = (enc_tok_attention_mask == 0)
        doc_pad_mask = (~enc_doc_mask)

        for layer in self.decoder_layers:
            x = layer(
                tgt_states=x,
                doc_states=enc_doc,
                token_states=enc_tok,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                doc_key_padding_mask=doc_pad_mask,
                memory_key_padding_mask=memory_pad_mask,
            )

        x = self.final_ln(x)
        return self.lm_head(x)

    def forward(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        doc_mask: torch.Tensor,
        doc_ranks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, K, L = src_input_ids.shape
        _, _, T = tgt_input_ids.shape

        enc_tok, enc_doc, enc_doc_mask = self.encode(
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            doc_mask=doc_mask,
            doc_ranks=doc_ranks,
        )

        flat_tgt = tgt_input_ids.view(B * K, T)
        flat_src_mask = src_attention_mask.view(B * K, L)

        logits = self.decode(
            decoder_input_ids=flat_tgt,
            enc_tok=enc_tok,
            enc_doc=enc_doc,
            enc_tok_attention_mask=flat_src_mask,
            enc_doc_mask=enc_doc_mask,
        )
        return logits.view(B, K, T, -1)

    def generate(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        doc_ranks: Optional[torch.Tensor] = None,
        max_length: int = 32,
    ) -> torch.Tensor:
        B, K, L = src_input_ids.shape
        device = src_input_ids.device

        enc_tok, enc_doc, enc_doc_mask = self.encode(
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            doc_mask=doc_mask,
            doc_ranks=doc_ranks,
        )
        flat_src_mask = src_attention_mask.view(B * K, L)

        ys = torch.full((B * K, 1), self.bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B * K, dtype=torch.bool, device=device)

        for _ in range(max_length):
            logits = self.decode(
                decoder_input_ids=ys,
                enc_tok=enc_tok,
                enc_doc=enc_doc,
                enc_tok_attention_mask=flat_src_mask,
                enc_doc_mask=enc_doc_mask,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            finished |= (next_token.squeeze(1) == self.eos_token_id)
            if finished.all():
                break

        return ys.view(B, K, -1)


# =========================
# Train / Eval
# =========================
def shift_right(tgt: torch.Tensor, bos_token_id: int, pad_token_id: int) -> torch.Tensor:
    decoder_input_ids = tgt.clone()
    decoder_input_ids[:, :, 1:] = tgt[:, :, :-1]
    decoder_input_ids[:, :, 0] = bos_token_id
    decoder_input_ids = decoder_input_ids.masked_fill(decoder_input_ids == -100, pad_token_id)
    return decoder_input_ids


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, leave=False):
        src_input_ids = batch["src_input_ids"].to(device)
        src_attention_mask = batch["src_attention_mask"].to(device)
        tgt_input_ids = batch["tgt_input_ids"].to(device)
        doc_mask = batch["doc_mask"].to(device)
        doc_ranks = batch["doc_ranks"].to(device)

        decoder_input_ids = shift_right(tgt_input_ids, model.bos_token_id, model.pad_token_id)
        logits = model(
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            tgt_input_ids=decoder_input_ids,
            doc_mask=doc_mask,
            doc_ranks=doc_ranks,
        )

        labels = tgt_input_ids.clone()
        labels[~doc_mask.unsqueeze(-1).expand_as(labels)] = model.pad_token_id

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            src_input_ids = batch["src_input_ids"].to(device)
            src_attention_mask = batch["src_attention_mask"].to(device)
            tgt_input_ids = batch["tgt_input_ids"].to(device)
            doc_mask = batch["doc_mask"].to(device)
            doc_ranks = batch["doc_ranks"].to(device)

            decoder_input_ids = shift_right(tgt_input_ids, model.bos_token_id, model.pad_token_id)
            logits = model(
                src_input_ids=src_input_ids,
                src_attention_mask=src_attention_mask,
                tgt_input_ids=decoder_input_ids,
                doc_mask=doc_mask,
                doc_ranks=doc_ranks,
            )

            labels = tgt_input_ids.clone()
            labels[~doc_mask.unsqueeze(-1).expand_as(labels)] = model.pad_token_id

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def generate_predictions(model, dataloader, tokenizer, device, max_target_len=32):
    model.eval()
    refs, preds = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            src_input_ids = batch["src_input_ids"].to(device)
            src_attention_mask = batch["src_attention_mask"].to(device)
            tgt_input_ids = batch["tgt_input_ids"].to(device)
            doc_mask = batch["doc_mask"].to(device)
            doc_ranks = batch["doc_ranks"].to(device)
            joint_refs = batch["joint_reference"]

            generated = model.generate(
                src_input_ids=src_input_ids,
                src_attention_mask=src_attention_mask,
                doc_mask=doc_mask,
                doc_ranks=doc_ranks,
                max_length=max_target_len,
            )

            B, K, _ = generated.shape
            for i in range(B):
                doc_preds = []
                for j in range(K):
                    if not bool(doc_mask[i, j].item()):
                        continue
                    pred_text = tokenizer.decode(generated[i, j], skip_special_tokens=True).strip()
                    doc_preds.append(pred_text)

                # final prediction follows your pipeline: explanation1 and explanation2
                joint_pred = " and ".join([p for p in doc_preds if p])
                preds.append(joint_pred.strip())
                refs.append(str(joint_refs[i]).strip())

    return refs, preds


# =========================
# Load dataset
# =========================
raw = datasets.load_dataset(args.hf_dataset) # type: ignore

train_df = pd.DataFrame(raw["train"])

# split train into train/val
val_size = int(0.1 * len(train_df))
val_df = train_df.sample(n=val_size, random_state=args.seed).reset_index(drop=True)
train_df = train_df.drop(val_df.index).reset_index(drop=True)
test_df = pd.DataFrame(raw["test"])
test_df = test_df[:args.test_size]

#if args.test_size < len(test_df):
#    test_df = test_df.sample(n=args.test_size, random_state=args.seed).reset_index(drop=True)

# small subset for quick tuning runs on local Machine
# train_df = train_df[:100]
# val_df = val_df[:20]
# test_df = test_df[:20]    

print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
print("Columns:", list(train_df.columns))
print("Expected format: query, doc1, doc2, explanation1, explanation2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if args.backbone == 'bart':
    tokenizer_name = args.bart_name
else:
    tokenizer_name = args.bert_name

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.sep_token


# =========================
# Hyperparameter tuning
# =========================
search_space = []
for bs in args.batch_sizes:
    for ep in args.epochs_list:
        for dr in args.dropouts:
            for wu in args.warmup_steps_list:
                search_space.append((bs, ep, dr, wu))

global_best = {
    "val_loss": float("inf"),
    "config": None,
    "model_path": None,
}
all_runs = []

for run_id, (batch_size, num_epochs, dropout, warmup_steps) in enumerate(search_space, start=1):
    print("\n" + "=" * 90)
    print(f"Run {run_id}/{len(search_space)}")
    print(f"batch_size={batch_size}, epochs={num_epochs}, dropout={dropout}, warmup_steps={warmup_steps}")

    train_dataset = LiEGESerpDataset(
        train_df, tokenizer,
        max_len=args.max_len,
        max_target_len=args.target_len,
        max_docs=2,
        use_joint_target=args.use_joint_target,
    )
    val_dataset = LiEGESerpDataset(
        val_df, tokenizer,
        max_len=args.max_len,
        max_target_len=args.target_len,
        max_docs=2,
        use_joint_target=args.use_joint_target,
    )
    test_dataset = LiEGESerpDataset(
        test_df, tokenizer, # type: ignore
        max_len=args.max_len,
        max_target_len=args.target_len,
        max_docs=2,
        use_joint_target=args.use_joint_target,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LiEGeFromScratch(
        backbone_name=args.bart_name if args.backbone == 'bart' else args.bert_name,
        tokenizer=tokenizer,
        backbone_type=args.backbone,
        num_local_layers=args.num_local_layers,
        num_global_layers=args.num_global_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_global_heads=args.num_global_heads,
        dropout=dropout,
        max_docs=args.max_docs,
        pooling=args.pooling,
        use_ordinal=args.use_ordinal,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    total_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{num_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        improved = early_stopper.step(val_loss)
        if improved:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())

        if early_stopper.should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    if best_state_dict is None:
        best_state_dict = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)

    run_model_path = os.path.join(
        args.output_dir,
        f"liege_{args.backbone}_best_model_run{run_id}_bs{batch_size}_ep{num_epochs}_do{dropout}_wu{warmup_steps}.pt"
    )
    torch.save(best_state_dict, run_model_path)

    refs, preds = generate_predictions(
        model, test_loader, tokenizer, device, max_target_len=args.target_len
    )

    run_output_csv = os.path.join(
        args.output_dir,
        f"liege_{args.backbone}_predictions_run{run_id}_bs{batch_size}_ep{num_epochs}_do{dropout}_wu{warmup_steps}.csv"
    )
    pd.DataFrame({
        "reference": refs,
        "prediction": preds
    }).to_csv(run_output_csv, index=False)

    print(f"Predictions saved to: {run_output_csv}")
    print(f"Model saved to: {run_model_path}")

    metrics = compute_metrics(refs, preds)

    run_result = {
        "run_id": run_id,
        "backbone": args.backbone,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "dropout": dropout,
        "warmup_steps": warmup_steps,
        "best_val_loss": best_val_loss,
        "model_path": run_model_path,
    }

    if isinstance(metrics, dict):
        run_result.update(metrics)

    all_runs.append(run_result)

    if best_val_loss < global_best["val_loss"]:
        global_best["val_loss"] = best_val_loss
        global_best["config"] = run_result
        global_best["model_path"] = run_model_path

summary_path = os.path.join(args.output_dir, f"liege_{args.backbone}_tuning_summary.csv")
pd.DataFrame(all_runs).sort_values("best_val_loss").to_csv(summary_path, index=False)

print("\n" + "=" * 90)
print("Best configuration:")
print(global_best["config"])
print(f"Summary saved to: {summary_path}")
print(f"Best model saved to: {global_best['model_path']}")
