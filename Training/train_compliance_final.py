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
import re
import sys

SEQ_LEN_A = 49_152
TARGET_LEN_A = 384  

SEQ_LEN_B = 196_608
TARGET_LEN_B = 896  

MAX_EPOCHS = 10
PAIRS_PER_EPOCH = 20000 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/workspace/geuvadis_full/output"
GTF_PATH = "gencode.v44.annotation.gtf" 
PRETRAINED_PATH = "./enformer_weights"
SAVE_DIR = "checkpoints_compliance"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f">>> OFFICIAL ARCHITECTURE REPLICATION | Device: {DEVICE}")

class DataManager:
    def __init__(self, root_dir, gtf_path):
        print(">>> Loading Data Manager...")
        self.base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.samples = {} 
        
        fasta_files = glob.glob(os.path.join(root_dir, "**", "*_personalized.fa"), recursive=True)
        quant_files = glob.glob(os.path.join(root_dir, "**", "quant.sf"), recursive=True)
        
        quant_map = {}
        for q in quant_files:
            parent = os.path.basename(os.path.dirname(q))
            sid_match = re.search(r'(HG\d+|NA\d+)', parent)
            if sid_match: quant_map[sid_match.group(1)] = q
        
        for f in fasta_files:
            fname = os.path.basename(f)
            match = re.search(r'(HG\d+|NA\d+)_(chr[\dXYM]+|[\dXYM]+)', fname)
            if match:
                sid, chrom = match.groups()
                if not chrom.startswith('chr'): chrom = 'chr' + chrom
                if sid in quant_map:
                    if sid not in self.samples: self.samples[sid] = {'quant': quant_map[sid]}
                    self.samples[sid][chrom] = f

        self.sample_ids = list(self.samples.keys())
        print(f"   Indexed {len(self.sample_ids)} valid samples.")

        print("   Caching Expression Data...")
        self.raw_cache = {}
        for sid in self.sample_ids:
             try:
                df = pd.read_csv(self.samples[sid]['quant'], sep="\t")
                self.raw_cache[sid] = dict(zip(df['Name'], df['TPM']))
             except: pass

        print("   Parsing GTF...")
        self.genes = []
        self.z_stats = {}
        
        all_transcripts = []
        with open(gtf_path) as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split('\t')
                if parts[2] == 'transcript':
                    try:
                        chrom = parts[0]
                        if not chrom.startswith('chr'): chrom = 'chr' + chrom
                        if '_' in chrom: continue 
                        tid = [x for x in parts[8].split(';') if 'transcript_id' in x][0].split('"')[1]
                        start, end = int(parts[3]), int(parts[4])
                        tss = start if parts[6] == '+' else end
                        all_transcripts.append({'id': tid, 'tss': tss, 'chrom': chrom})
                    except: continue
        
        unique_transcripts = []
        seen_tss = set()
        for t in all_transcripts:
            key = (t['chrom'], t['tss'])
            if key not in seen_tss:
                unique_transcripts.append(t)
                seen_tss.add(key)
        
        print("   Calculating Whole-Population Z-Scores...")
        for g in unique_transcripts:
            vals = []
            for sid in self.sample_ids:
                if sid in self.raw_cache and g['id'] in self.raw_cache[sid]:
                    vals.append(self.raw_cache[sid][g['id']])
            
            if len(vals) > 50: 
                vals = np.log1p(np.array(vals))
                std = np.std(vals)
                if std < 1e-4: continue
                mean = np.mean(vals)
                self.z_stats[g['id']] = {'mean': mean, 'std': std}
                self.genes.append(g)
                
        print(f"   Final Valid Genes: {len(self.genes)}")

    def get_dataset(self, indices, is_train, seq_len):
        return RobustDataset(self, [self.genes[i] for i in indices], is_train, seq_len)

class RobustDataset(Dataset):
    def __init__(self, manager, genes, is_train, seq_len):
        self.manager = manager
        self.genes = genes
        self.is_train = is_train
        self.seq_len = seq_len
        self.generate_epoch_pairs()

    def generate_epoch_pairs(self):
        self.pairs = []
        target = PAIRS_PER_EPOCH if self.is_train else PAIRS_PER_EPOCH // 5
        attempts = 0
        while len(self.pairs) < target and attempts < target * 5:
            attempts += 1
            g = random.choice(self.genes)
            s1, s2 = random.sample(self.manager.sample_ids, 2)
            chrom = g['chrom']
            
            if chrom in self.manager.samples[s1] and chrom in self.manager.samples[s2]:
                if g['id'] in self.manager.raw_cache[s1] and g['id'] in self.manager.raw_cache[s2]:
                    raw1 = np.log1p(self.manager.raw_cache[s1][g['id']])
                    raw2 = np.log1p(self.manager.raw_cache[s2][g['id']])
                    stats = self.manager.z_stats[g['id']]
                    
                    z1 = (raw1 - stats['mean']) / stats['std']
                    z2 = (raw2 - stats['mean']) / stats['std']
                    
                    if abs(z1 - z2) > 20: continue 
                    
                    self.pairs.append({
                        'tss': g['tss'], 'chrom': chrom, 's1': s1, 's2': s2,
                        'z_diff': z1 - z2
                    })

    def __len__(self): return len(self.pairs)

    def load_seq(self, sid, chrom, tss):
        extractor = kipoiseq.extractors.FastaStringExtractor(self.manager.samples[sid][chrom])
        f_chrom = list(extractor.fasta.keys())[0]
        
        shift = random.randint(-3, 3) if self.is_train else 0
        center = tss + shift
        interval = Interval(f_chrom, center - self.seq_len // 2, center + self.seq_len // 2)
        
        try: seq = extractor.extract(interval)
        except: seq = "N" * self.seq_len
        
        if len(seq) < self.seq_len: seq += "N" * (self.seq_len - len(seq))
        elif len(seq) > self.seq_len: seq = seq[:self.seq_len]
        
        if self.is_train and random.random() > 0.5:
            seq = seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]
            
        arr = np.zeros((len(seq), 4), dtype=np.float32)
        for i, char in enumerate(seq.upper()):
            if char in self.manager.base_map: arr[i, self.manager.base_map[char]] = 1.0
        return arr

    def __getitem__(self, idx):
        p = self.pairs[idx]
        x1 = self.load_seq(p['s1'], p['chrom'], p['tss'])
        x2 = self.load_seq(p['s2'], p['chrom'], p['tss'])
        return torch.tensor(x1), torch.tensor(x2), torch.tensor([p['z_diff']], dtype=torch.float32)

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class EnformerWrapper(nn.Module):
    def __init__(self, mode="Original"):
        super().__init__()
        self.mode = mode
        dim = 5313
        
        if mode == "Original":
            target_len = TARGET_LEN_A
        else:
            target_len = TARGET_LEN_B
        self.backbone = Enformer.from_pretrained(
            PRETRAINED_PATH, 
            use_checkpointing=True, 
            target_length=target_len
        )
        
        for param in self.backbone.parameters(): param.requires_grad = True
        
        if mode == "Original":
            self.attn_pool = nn.Linear(dim, 1)
            self.final = nn.Linear(dim, 1)
            
        elif mode == "SwiGLU":
            self.norm = nn.LayerNorm(dim)
            self.project = nn.Linear(dim, dim * 2)
            self.activation = SwiGLU()
            self.dropout = nn.Dropout(0.1) 
            self.attn_pool = nn.Linear(dim, 1)
            self.final = nn.Linear(dim, 1)

    def forward_single(self, x):
        out = self.backbone(x)['human'] 
        
        if self.mode == "Original":
            n_bins = out.shape[1]
            center = n_bins // 2
            out = out[:, center-5:center+5, :]
            
            w = F.softmax(self.attn_pool(out), dim=1)
            pooled = (out * w).sum(dim=1)
            return self.final(pooled)
            
        elif self.mode == "SwiGLU":
            mid = out.shape[1] // 2
            out = out[:, mid-2:mid+2, :] 
            
            out = self.norm(out)
            out = self.dropout(self.activation(self.project(out)))
            w = F.softmax(self.attn_pool(out), dim=1)
            pooled = (out * w).sum(dim=1)
            return self.final(pooled)

    def forward(self, x1, x2):
        return self.forward_single(x1) - self.forward_single(x2)

def train_cycle(model, train_dl, val_dl, epochs, model_type):
    if model_type == "Original":
        lr, wd, crit = 1e-4, 1e-3, nn.MSELoss()
    else:
        lr, wd, crit = 2e-5, 1e-2, nn.HuberLoss(delta=1.0)
        
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_v = float('inf')
    
    for ep in range(epochs):
        model.train()
        losses = []
        opt.zero_grad() 
        for i, (x1, x2, y) in enumerate(train_dl):
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
            pred = model(x1, x2)
            loss = crit(pred, y)
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                opt.zero_grad()
            losses.append(loss.item() * ACCUMULATION_STEPS)
            
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x1, x2, y in val_dl:
                val_losses.append(crit(model(x1.to(DEVICE), x2.to(DEVICE)), y.to(DEVICE)).item())
        
        avg_v = np.mean(val_losses)
        print(f"   Ep {ep+1}: Train={np.mean(losses):.4f} | Val={avg_v:.4f}")
        if avg_v < best_v: best_v = avg_v
        
        train_dl.dataset.generate_epoch_pairs()
        val_dl.dataset.generate_epoch_pairs()
    return best_v

def main():
    manager = DataManager(DATA_ROOT, GTF_PATH)
    all_indices = list(range(len(manager.genes)))
    
    print("\n" + "="*50)
    print(f"[MODEL A] Paper Copy: {SEQ_LEN_A}bp | MSE | Unfrozen")
    print("="*50)
    t_idx, v_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
    ds_train = manager.get_dataset(t_idx, True, SEQ_LEN_A)
    ds_val = manager.get_dataset(v_idx, False, SEQ_LEN_A)
    
    model_a = EnformerWrapper(mode="Original").to(DEVICE)
    score_a = train_cycle(model_a, 
                          DataLoader(ds_train, batch_size=1, shuffle=True), 
                          DataLoader(ds_val, batch_size=1), 
                          MAX_EPOCHS, "Original")
    print(f"Model A Final Score (MSE): {score_a:.5f}")

    
    print("\n" + "="*50)
    print(f"[MODEL B] Enhanced: {SEQ_LEN_B}bp | SwiGLU | Huber")
    print("="*50)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (ti, vi) in enumerate(kf.split(all_indices)):
        print(f"   >>> Fold {fold+1}/5...")
        ds_train = manager.get_dataset(ti, True, SEQ_LEN_B) 
        ds_val = manager.get_dataset(vi, False, SEQ_LEN_B)
        
        model_b = EnformerWrapper(mode="SwiGLU").to(DEVICE)
        score = train_cycle(model_b, 
                            DataLoader(ds_train, batch_size=1, shuffle=True), 
                            DataLoader(ds_val, batch_size=1), 
                            MAX_EPOCHS, "SwiGLU")
        fold_scores.append(score)
        print(f"        Fold Score: {score:.5f}")
    
    print(f"\nFINAL SUMMARY:\nModel A (MSE): {score_a:.5f}\nModel B (Huber): {np.mean(fold_scores):.5f}")

if __name__ == "__main__": main()
