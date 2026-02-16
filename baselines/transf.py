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

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, encoder_nhead=4, encoder_nlayers=12, 
                 decoder_nhead=4, decoder_nlayers=12, ff_dim=3072, max_len=512):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.pos_decoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model, encoder_nhead, ff_dim, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, decoder_nhead, ff_dim, batch_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, encoder_nlayers)
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_nlayers)

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src_input_ids, tgt_input_ids, src_mask=None, tgt_mask=None):
        src_emb = self.pos_encoder(self.embedding(src_input_ids))
        tgt_emb = self.pos_decoder(self.embedding(tgt_input_ids))

        src_emb = self.embedding(src_input_ids)
        tgt_emb = self.embedding(tgt_input_ids)

        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
        return self.output_layer(output)
    
class TransformerDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_len=512, max_target_len=32):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src = self.tokenizer.encode(
            self.inputs[idx], 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len
        )
        
        tgt = self.tokenizer.encode(
            self.targets[idx], 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_target_len
        )
        
        return torch.tensor(src), torch.tensor(tgt)

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

test_df = test_df.sample(n=args.test_size, random_state=42).reset_index(drop=True)

print(len(train_df), len(val_df), len(test_df))

train_src = train_df["query"]+"[SEP]"+train_df["doc"].tolist()
train_tgt = train_df["explanation"].tolist()

val_src = val_df["query"]+"[SEP]"+val_df["doc"].tolist()
val_tgt = val_df["explanation"].tolist()

test_src = test_df["query"]+"[SEP]"+test_df["doc"].tolist()
test_tgt = test_df["explanation"].tolist()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

tokenizer_name = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

train_dataset = TransformerDataset(train_src, train_tgt, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
val_dataset = TransformerDataset(val_src, val_tgt, tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
test_dataset = TransformerDataset(test_src, test_tgt, tokenizer)
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
    for src, tgt in tqdm(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        logits = model(src, tgt_input, src_mask=None, tgt_mask=tgt_mask)
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
        for src, tgt in tqdm(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            logits = model(src, tgt_input, src_mask=None, tgt_mask=tgt_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_labels.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)

def generates(model, dataloader, tokenizer, device, max_target_len=10):
    model.eval()
    ref = []
    pred = []
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            src_mask = create_src_mask(src, tokenizer.pad_token_id).to(device)

            memory = model.encoder(model.embedding(src), src_key_padding_mask=src_mask)
            ys = torch.full((src.size(0), 1), tokenizer.cls_token_id, dtype=torch.long).to(device)
            
            for i in range(max_target_len):
                tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
                out = model.decoder(model.embedding(ys), memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
                logits = model.output_layer(out[:, -1, :])
                next_token = logits.argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)

                if (next_token.squeeze(-1) == tokenizer.sep_token_id).any().item():                           
                    break
                
            output = ys[:, 1:]
            for j in range(output.size(0)):
                gen_text = tokenizer.decode(output[j], skip_special_tokens=True)
                label = tokenizer.decode(tgt[j], skip_special_tokens=True)
                ref.append(label)
                pred.append(gen_text)
    
    return ref, pred

# Train
for epoch in range(args.epoch):
    train_loss = train(model, train_dataloader, optimizer, criterion, device)
    val_loss = validate(model, val_dataloader, criterion, device)
    print("")
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

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

