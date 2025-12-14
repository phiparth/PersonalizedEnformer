#!/bin/bash

set -e
DATA_ROOT="/workspace/geuvadis_check/output"
WORKDIR="/workspace/final_robust_training"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

CONDA_BASE=$(conda info --base)
ENV_PATH="$CONDA_BASE/envs/enformer_stable"
PYTHON_EXEC="$ENV_PATH/bin/python"

echo ">>> [FIX] Pre-downloading Enformer weights to local folder..."
mkdir -p enformer_weights
cd enformer_weights

if [ ! -f "config.json" ]; then
    wget -q -nc https://huggingface.co/EleutherAI/enformer-official-rough/resolve/main/config.json
fi

if [ ! -f "pytorch_model.bin" ]; then
    wget -q -nc https://huggingface.co/EleutherAI/enformer-official-rough/resolve/main/pytorch_model.bin
fi

cd "$WORKDIR"
echo ">>> Weights downloaded successfully."
cat << 'EOF' > train_paper_robust.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from enformer_pytorch import Enformer
import kipoiseq
from kipoiseq import Interval
from sklearn.model_selection import KFold, train_test_split
import os
import glob
import random
import sys

SEQ_LENGTH = 196_608 
BATCH_SIZE = 1 
LEARNING_RATE = 5e-5      
MAX_EPOCHS = 10           
PAIRS_PER_EPOCH = 20000 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/workspace/geuvadis_check/output"
GTF_PATH = "gencode.v44.annotation.gtf"
SAVE_DIR = "checkpoints"
PRETRAINED_PATH = "./enformer_weights" 

os.makedirs(SAVE_DIR, exist_ok=True)

print(f">>> FINAL EXPERIMENT INIT | Device: {DEVICE}")

class DataManager:
    def __init__(self, root_dir, gtf_path):
        print(">>> Loading Data Manager...")
        self.base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        self.samples = {} 
        fasta_files = glob.glob(os.path.join(root_dir, "*_chr22_personalized.fa"))
        for f in fasta_files:
            sid = os.path.basename(f).split('_')[0]
            qpath = os.path.join(root_dir, f"{sid}_quant", "quant.sf")
            if os.path.exists(qpath):
                self.samples[sid] = {'fasta': f, 'quant': qpath}
        self.sample_ids = list(self.samples.keys())
        print(f"   Samples: {len(self.sample_ids)}")

        print("   Caching Expression...")
        self.raw_cache = {}
        for sid in self.sample_ids:
             try:
                df = pd.read_csv(self.samples[sid]['quant'], sep="\t")
                self.raw_cache[sid] = dict(zip(df['Name'], df['TPM']))
             except: pass

        print("   Parsing Genes & Calculating Stats...")
        self.genes = []
        self.z_stats = {}
        
        all_transcripts = []
        with open(gtf_path) as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split('\t')
                if parts[0] not in ['22', 'chr22']: continue
                if parts[2] == 'transcript':
                    try:
                        tid = [x for x in parts[8].split(';') if 'transcript_id' in x][0].split('"')[1]
                        start, end = int(parts[3]), int(parts[4])
                        tss = start if parts[6] == '+' else end
                        all_transcripts.append({'id': tid, 'tss': tss})
                    except: continue
        
        sample_subset = self.sample_ids[:50]
        for g in all_transcripts:
            vals = []
            for sid in sample_subset:
                if sid in self.raw_cache and g['id'] in self.raw_cache[sid]:
                    vals.append(self.raw_cache[sid][g['id']])
            
            if len(vals) > 5: 
                vals = np.log1p(np.array(vals))
                mean = np.mean(vals)
                std = np.std(vals) + 1e-6
                self.z_stats[g['id']] = {'mean': mean, 'std': std}
                self.genes.append(g)
                
        print(f"   Valid Genes: {len(self.genes)}")

    def get_dataset(self, gene_indices, is_train=True):
        genes_subset = [self.genes[i] for i in gene_indices]
        return RobustDataset(self, genes_subset, is_train)

class RobustDataset(Dataset):
    def __init__(self, manager, genes, is_train):
        self.manager = manager
        self.genes = genes
        self.is_train = is_train
        self.generate_epoch_pairs()
        
    def generate_epoch_pairs(self):
        self.pairs = []
        count = PAIRS_PER_EPOCH if self.is_train else PAIRS_PER_EPOCH // 5
        attempts = 0
        while len(self.pairs) < count and attempts < count * 5:
            attempts += 1
            g = random.choice(self.genes)
            s1, s2 = random.sample(self.manager.sample_ids, 2)
            
            if g['id'] in self.manager.raw_cache.get(s1, {}) and g['id'] in self.manager.raw_cache.get(s2, {}):
                raw1 = np.log1p(self.manager.raw_cache[s1][g['id']])
                raw2 = np.log1p(self.manager.raw_cache[s2][g['id']])
                stats = self.manager.z_stats[g['id']]
                z1 = (raw1 - stats['mean']) / stats['std']
                z2 = (raw2 - stats['mean']) / stats['std']
                self.pairs.append({'tss': g['tss'], 's1': s1, 's2': s2, 'z_diff': z1 - z2})

    def __len__(self): return len(self.pairs)

    def load_seq(self, fasta_path, tss):
        extractor = kipoiseq.extractors.FastaStringExtractor(fasta_path)
        chrom = list(extractor.fasta.keys())[0]
        
        shift = random.randint(-2, 2) if self.is_train else 0
        center = tss + shift
        interval = Interval(chrom, center - SEQ_LENGTH // 2, center + SEQ_LENGTH // 2)
        try: seq = extractor.extract(interval)
        except: seq = "N" * SEQ_LENGTH
        
        if len(seq) < SEQ_LENGTH: seq += "N" * (SEQ_LENGTH - len(seq))
        elif len(seq) > SEQ_LENGTH: seq = seq[:SEQ_LENGTH]
        
        if self.is_train and random.random() > 0.5:
            trans = str.maketrans("ACGTNacgtn", "TGCANtgcan")
            seq = seq.translate(trans)[::-1]
            
        arr = np.zeros((len(seq), 4), dtype=np.float32)
        for i, char in enumerate(seq.upper()):
            if char in self.manager.base_map: 
                arr[i, self.manager.base_map[char]] = 1.0
        return arr

    def __getitem__(self, idx):
        p = self.pairs[idx]
        x1 = self.load_seq(self.manager.samples[p['s1']]['fasta'], p['tss'])
        x2 = self.load_seq(self.manager.samples[p['s2']]['fasta'], p['tss'])
        return (torch.tensor(x1, dtype=torch.float32), 
                torch.tensor(x2, dtype=torch.float32), 
                torch.tensor([p['z_diff']], dtype=torch.float32))

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class EnformerRobust(nn.Module):
    def __init__(self, mode="Original"):
        super().__init__()
        self.backbone = Enformer.from_pretrained(PRETRAINED_PATH)
        for param in self.backbone.parameters(): param.requires_grad = False
        for param in self.backbone.transformer[-1:].parameters(): param.requires_grad = True
        
        dim = 5313 
        
        if mode == "SwiGLU":
            self.norm = nn.LayerNorm(dim)
            self.project = nn.Linear(dim, dim * 2)
            self.activation = SwiGLU()
        else: # Original
            self.norm = nn.Identity()
            self.project = nn.Linear(dim, dim)
            self.activation = nn.GELU()
            
        self.attn_pool = nn.Linear(dim, 1)
        self.final = nn.Linear(dim, 1)

    def forward_single(self, x):
        out = self.backbone(x)['human']
        mid = out.shape[1] // 2
        out = out[:, mid-2:mid+2, :] 
        out = self.norm(out)
        out = self.activation(self.project(out))
        weights = F.softmax(self.attn_pool(out), dim=1)
        pooled = (out * weights).sum(dim=1)
        return self.final(pooled)

    def forward(self, x1, x2):
        return self.forward_single(x1) - self.forward_single(x2)

def train_cycle(model, train_dl, val_dl, epochs=5):
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    crit = nn.HuberLoss(delta=1.0)
    
    best_val = float('inf')
    
    for ep in range(epochs):
        model.train()
        losses = []
        for x1, x2, target in train_dl:
            x1, x2, target = x1.to(DEVICE), x2.to(DEVICE), target.to(DEVICE)
            opt.zero_grad()
            pred = model(x1, x2)
            loss = crit(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) 
            opt.step()
            losses.append(loss.item())
            
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x1, x2, target in val_dl:
                x1, x2, target = x1.to(DEVICE), x2.to(DEVICE), target.to(DEVICE)
                val_losses.append(crit(model(x1, x2), target).item())
        
        avg_v = np.mean(val_losses)
        print(f"   Ep {ep+1}: Train={np.mean(losses):.4f} | Val={avg_v:.4f}")
        
        if avg_v < best_val: best_val = avg_v
        
        # Refresh Data
        train_dl.dataset.generate_epoch_pairs()
        val_dl.dataset.generate_epoch_pairs()
        
    return best_val

def main():
    manager = DataManager(DATA_ROOT, GTF_PATH)
    all_indices = list(range(len(manager.genes)))
    
    print("\n" + "="*40)
    print("[MODEL A] Original | Fixed 80/20 Split")
    print("="*40)
    
    train_idx, val_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
    ds_train = manager.get_dataset(train_idx, is_train=True)
    ds_val = manager.get_dataset(val_idx, is_train=False)
    
    model_a = EnformerRobust(mode="Original").to(DEVICE)
    score_a = train_cycle(model_a, DataLoader(ds_train, batch_size=1, shuffle=True), 
                          DataLoader(ds_val, batch_size=1), epochs=MAX_EPOCHS)
    print(f"Model A Score: {score_a:.5f}")

    print("\n" + "="*40)
    print("[MODEL B] SwiGLU | 5-Fold Cross Validation")
    print("="*40)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_indices)):
        print(f"   >>> Running Fold {fold+1}/5...")
        ds_train = manager.get_dataset(train_idx, is_train=True)
        ds_val = manager.get_dataset(val_idx, is_train=False)
        
        model_b = EnformerRobust(mode="SwiGLU").to(DEVICE)
        score = train_cycle(model_b, DataLoader(ds_train, batch_size=1, shuffle=True), 
                            DataLoader(ds_val, batch_size=1), epochs=MAX_EPOCHS)
        fold_scores.append(score)
        print(f"       Fold {fold+1} Score: {score:.5f}")
        
    score_b = np.mean(fold_scores)
    
    print("\n" + "#"*40)
    print(" FINAL RESULTS (Huber Loss)")
    print("#"*40)
    print(f"Model A (Original, Fixed): {score_a:.5f}")
    print(f"Model B (SwiGLU, 5-Fold):  {score_b:.5f}")
    
    if score_b < score_a:
        print("WINNER: Model B (SwiGLU)")
    else:
        print("WINNER: Model A (Original)")

if __name__ == "__main__":
    main()
EOF

echo ">>> Launching Final Robust Training..."
nohup "$PYTHON_EXEC" -u train_paper_robust.py > final_results.log 2>&1 &
echo ">>> RUNNING. Track progress with:"
echo "    tail -f final_results.log"
