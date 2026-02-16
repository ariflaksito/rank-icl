# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

from transformers import BertTokenizerFast, logging
logging.set_verbosity_error()

import math
import torch  
import argparse
from tqdm import tqdm
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset 
from utils.a5evaluates import compute_metrics
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="use --help for investigating input params")
parser.add_argument('--hf-dataset', type=str, required=True, help="Huggingface dataset name")
parser.add_argument('--test-size', type=int, required=True, help="Train size data, e.g., 2000")
parser.add_argument('--epoch', type=int, required=True, help="Number of training epochs, e.g., 3")
parser.add_argument('--target-len', type=int, required=True, help="Max target length for generation, e.g., 50")
parser.add_argument('--batch', type=int, required=True, help="Batch size for training, e.g., 8")
parser.add_argument('--lr', type=float, required=True, help="Learning rate for optimizer, e.g., 1e-5")
parser.add_argument('--output', type=str, required=True, help="full path of generating output explanations")

args = parser.parse_args()
print("All arguments passed:", flush=True)
print(vars(args), flush=True)

"""# Transformer modified class"""

# Positional Encoding is used for Transformer models
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class QueryAttentionLayer(nn.Module):
    def __init__(self, d_model, query_dim):
        super().__init__()
        # Parameter matrices
        self.WQ = nn.Linear(d_model, query_dim)
        self.WK = nn.Linear(d_model * 2, d_model)
        self.WV = nn.Linear(d_model * 2, d_model)

    def forward(self, zd, zq):
        # Q = zd * WQ
        Q = self.WQ(zd)

        # K = (zq || zd) * WK
        K = self.WK(torch.cat((zq, zd), dim=-1))

        # V = (zq || zd) * WV
        V = self.WV(torch.cat((zq, zd), dim=-1))

        # Attention: softmax(QK^T / sqrt(K)) * V
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, encoder_nhead=4, encoder_nlayers=12,
                 decoder_nhead=4, decoder_nlayers=12, ff_dim=3072, max_len=512, query_dim=768, pad_token_id=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.pos_decoder = PositionalEncoding(d_model, max_len)
        self.pad_token_id = pad_token_id

        self.query_attention = QueryAttentionLayer(d_model, query_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model, encoder_nhead, ff_dim, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, decoder_nhead, ff_dim, batch_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, encoder_nlayers)
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_nlayers)

        self.output_layer = nn.Linear(d_model, vocab_size)

    def create_query_mask(self, src_query_ids, src_doc_ids):
        """
        Create a combined padding mask for the concatenated query and document memory.
        True indicates a padded position that should be ignored by attention.
        """
        query_padding_mask = (src_query_ids == self.pad_token_id)
        doc_padding_mask = (src_doc_ids == self.pad_token_id)
        combined_padding_mask = torch.cat((query_padding_mask, doc_padding_mask), dim=1)
        return combined_padding_mask.to(src_query_ids.device)

    def forward(self, src_query_ids, src_doc_ids, tgt_input_ids, tgt_mask=None):
        # Apply embeddings and positional encoding
        src_query_emb = self.pos_encoder(self.embedding(src_query_ids))  # Query Embedding
        src_doc_emb = self.pos_encoder(self.embedding(src_doc_ids))  # Document Embedding
        tgt_emb = self.pos_decoder(self.embedding(tgt_input_ids))  # Target Embedding

        # Create padding mask for src_query_ids
        src_query_padding_mask = (src_query_ids == self.pad_token_id).to(src_query_ids.device)

        # Pass the query embedding through the encoder to get memory specific to the query
        memory_query = self.encoder(src_query_emb, src_key_padding_mask=src_query_padding_mask)

        # Apply the query-focused attention:
        # Here, `zd` is the sequence to be 'focused' (document), `zq` is the 'query' for focusing (encoded query).
        # The QueryAttentionLayer returns a sequence with the length of `zd` (src_doc_emb).
        query_focused_doc = self.query_attention(src_doc_emb, memory_query)

        # Concatenate the encoded query and the query-focused document to form the decoder's memory
        # This combined memory will have a sequence length of 512 (from query) + 512 (from doc) = 1024.
        combined_memory_for_decoder = torch.cat((memory_query, query_focused_doc), dim=1)

        # Create a mask for the combined memory. This mask will have length 1024.
        memory_key_padding_mask = self.create_query_mask(src_query_ids, src_doc_ids)

        # Pass the combined memory and its corresponding mask to the decoder
        output = self.decoder(tgt_emb, combined_memory_for_decoder, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)

        return self.output_layer(output)

"""# Data Loader"""

class TransformerDataset(Dataset):
    def __init__(self, q, d, e, tokenizer, query_max_len=512, doc_max_len=512, max_target_len=args.target_len):
        self.q = q
        self.d = d
        self.e = e
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.doc_max_len = doc_max_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        src1 = self.tokenizer.encode(
            self.q[idx],
            truncation=True,
            padding='max_length',
            max_length=self.query_max_len)

        src2 = self.tokenizer.encode(
            self.d[idx],
            truncation=True,
            padding='max_length',
            max_length=self.doc_max_len)

        tgt = self.tokenizer.encode(
            self.e[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_target_len)

        return torch.tensor(src1), torch.tensor(src2), torch.tensor(tgt)

"""# Load Dataset"""

import datasets
import pandas as pd

if args.hf_dataset == 'wiki':
    train_csv = "data/wikisa1/train.csv"
    test_csv = "data/wikisa1/test.csv"
elif args.hf_dataset == 'exa':
    train_csv = "data/exarank1/train.csv"
    test_csv = "data/exarank1/test.csv"

train_df = pd.read_csv(train_csv)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
test_df = pd.read_csv(test_csv)

test_df = test_df.sample(n=args.test_size, random_state=42)

print(len(train_df), len(val_df), len(test_df))

train_src1 = train_df["query"].tolist()
train_src2 = train_df["doc"].tolist()
train_tgt = train_df["explanation"].tolist()

val_src1 = val_df["query"].tolist()
val_src2 = val_df["doc"].tolist()
val_tgt = val_df["explanation"].tolist()

test_src1 = test_df["query"].tolist()
test_src2 = test_df["doc"].tolist()
test_tgt = test_df["explanation"].tolist()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

tokenizer_name = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

train_dataset = TransformerDataset(train_src1, train_src2, train_tgt, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
val_dataset = TransformerDataset(val_src1, val_src2, val_tgt, tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
test_dataset = TransformerDataset(test_src1, test_src2, test_tgt, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

"""# Train GenEx model"""

model = TransformerEncoderDecoder(vocab_size=tokenizer.vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def create_src_mask(src, pad_token_id):
    return (src == pad_token_id)

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src1, src2, tgt in tqdm(dataloader):
        src1, src2, tgt = src1.to(device), src2.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        logits = model(src1, src2, tgt_input, tgt_mask=tgt_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src1, src2, tgt in tqdm(dataloader):
            src1, src2, tgt = src1.to(device), src2.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            logits = model(src1, src2, tgt_input, tgt_mask=tgt_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_labels.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)

def generates(model, dataloader, tokenizer, device, max_target_len=args.target_len):
    model.eval()
    ref = []
    pred = []

    with torch.no_grad():
        for src1, src2, tgt in tqdm(dataloader):
            src1, src2, tgt = src1.to(device), src2.to(device), tgt.to(device)

            # Generate embeddings for src1 (query) and src2 (document)
            src_query_emb = model.embedding(src1)
            src_doc_emb = model.embedding(src2)

            # Create padding mask for the query input
            src_query_padding_mask = (src1 == model.pad_token_id).to(device)

            # Pass the query embedding through the encoder
            memory_query = model.encoder(src_query_emb, src_key_padding_mask=src_query_padding_mask)

            # Apply query-focused attention: zd (document) focused by zq (query)
            query_focused_doc = model.query_attention(src_doc_emb, memory_query)

            # Concatenate the query memory and the query-focused document for the decoder's memory
            combined_memory_for_decoder = torch.cat((memory_query, query_focused_doc), dim=1)

            # Create the combined memory key padding mask for the decoder
            memory_key_padding_mask_decoder = model.create_query_mask(src1, src2)

            # Initialize decoder input with CLS token
            ys = torch.full((src1.size(0), 1), tokenizer.cls_token_id  , dtype=torch.long).to(device)

            for i in range(max_target_len):
                tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
                out = model.decoder(model.embedding(ys), combined_memory_for_decoder,
                                    tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask_decoder)
                logits = model.output_layer(out[:, -1, :])
                next_token = logits.argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)

                # Stop generation if all sequences in the batch predict the SEP token
                if (next_token.squeeze(-1) == tokenizer.sep_token_id).all().item():
                    break

            # Decode generated output for the entire batch
            output = ys[:, 1:] # Remove the initial CLS token
            for j in range(output.size(0)): # Iterate through batch items
                gen_text = tokenizer.decode(output[j], skip_special_tokens=True)
                label = tokenizer.decode(tgt[j], skip_special_tokens=True) # Assuming tgt has original target for comparison
                ref.append(label)
                pred.append(gen_text)

    return ref, pred

# Train
for epoch in range(args.epoch):
    train_loss = train(model, train_dataloader, optimizer, criterion, device)
    val_loss = validate(model, val_dataloader, criterion, device)
    print(f"\nEpoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

ref, pred = generates(model, test_dataloader, tokenizer, device, max_target_len=args.target_len)

results = {
    "reference": ref,
    "prediction": pred
}
dfs = pd.DataFrame(results)

# Save the DataFrame to a new CSV file
dfs.to_csv(args.output, index=False)
print(f"Explanations generated and saved to: {args.output}", flush=True)

print(f"---", flush=True)
compute_metrics(ref, pred)

