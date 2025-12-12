cd /workspace/experiment_hg00152

cat << 'EOF' > experiment_comparison.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from enformer_pytorch import Enformer
from sklearn.model_selection import KFold, train_test_split
import kipoiseq
from kipoiseq import Interval
import os

# --- CONFIGURATION ---
SEQ_LENGTH = 196_608 
BATCH_SIZE = 1 
LEARNING_RATE = 1e-4  # Reduced slightly for stability
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATHS ---
FASTA_PATH = "/workspace/geuvadis/vcf/HG00152_chr22_personalized.fa"
TARGET_PATH = "/workspace/geuvadis/salmon_out/HG00152_quant/quant.sf"
GTF_PATH = "/workspace/geuvadis/reference/gencode.v44.annotation.gtf"

print(f">>> Running on: {DEVICE}")

class SingleSampleDataset(Dataset):
    def __init__(self, fasta_path, target_path, gtf_path):
        print(f"Loading data for HG00152...")
        try:
            quant_df = pd.read_csv(target_path, sep="\t")
            self.tpm_map = dict(zip(quant_df['Name'], quant_df['TPM']))
        except: sys.exit(1)
            
        if not os.path.exists(fasta_path): sys.exit(1)
        self.extractor = kipoiseq.extractors.FastaStringExtractor(fasta_path)
        self.base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.chrom_name = list(self.extractor.fasta.keys())[0]
        self.one_hot = kipoiseq.transforms.OneHot(alphabet="ACGT", neutral_value=0)
        
        self.samples = []
        with open(gtf_path) as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split('\t')
                if parts[0] not in ['22', 'chr22']: continue
                if parts[2] == 'transcript':
                    try:
                        tid = [x for x in parts[8].split(';') if 'transcript_id' in x][0].split('"')[1]
                        if tid in self.tpm_map:
                            val = np.log1p(self.tpm_map[tid])
                            start, end = int(parts[3]), int(parts[4])
                            tss = start if parts[6] == '+' else end
                            self.samples.append({'tss': tss, 'target': val})
                    except: continue
        print(f"Dataset Ready: Found {len(self.samples)} transcripts.")

    def __len__(self): return len(self.samples)

    def robust_one_hot(self, seq):
        arr = np.zeros((len(seq), 4), dtype=np.float32)
        for i, char in enumerate(seq.upper()):
            if char in self.base_map:
                arr[i, self.base_map[char]] = 1.0
        return arr

    def __getitem__(self, idx):
        item = self.samples[idx]
        interval = Interval(self.chrom_name, item['tss'] - SEQ_LENGTH // 2, item['tss'] + SEQ_LENGTH // 2)
        try: seq = self.extractor.extract(interval)
        except: seq = "N" * SEQ_LENGTH
        if len(seq) < SEQ_LENGTH: seq += "N" * (SEQ_LENGTH - len(seq))
        elif len(seq) > SEQ_LENGTH: seq = seq[:SEQ_LENGTH]
        seq_enc = self.robust_one_hot(seq)
        return torch.tensor(seq_enc, dtype=torch.float32), torch.tensor([item['target']], dtype=torch.float32)

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class EnformerFineTuner(nn.Module):
    def __init__(self, mode="Original"):
        super().__init__()
        self.backbone = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
        for param in self.backbone.parameters(): param.requires_grad = False
        
        dim = 5313 
        self.norm = nn.LayerNorm(dim)
        
        if mode == "SwiGLU":
            self.project = nn.Linear(dim, dim * 2)
            self.activation = SwiGLU()
        else:
            self.project = nn.Linear(dim, dim)
            self.activation = nn.GELU()
            
        self.attn_pool = nn.Linear(dim, 1)
        self.final_head = nn.Linear(dim, 1)

    def forward(self, x):
        out = self.backbone(x)['human']
        mid = out.shape[1] // 2
        out = out[:, mid-2:mid+2, :]
        
        out = self.norm(out)
        
        out = self.activation(self.project(out))
        weights = F.softmax(self.attn_pool(out), dim=1)
        pooled = (out * weights).sum(dim=1)
        return self.final_head(pooled)

def train_and_eval(model, train_loader, val_loader, epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    crit = nn.MSELoss()
    best_val = float('inf')
    
    for ep in range(epochs):
        model.train()
        for seq, y in train_loader:
            seq, y = seq.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(seq)
            loss = crit(pred, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for seq, y in val_loader:
                seq, y = seq.to(DEVICE), y.to(DEVICE)
                v_loss += crit(model(seq), y).item()
        
        avg_val = v_loss / len(val_loader)
        if avg_val < best_val: best_val = avg_val
    return best_val

# --- 4. MAIN ---
def main():
    print(">>> INITIALIZING EXPERIMENT ON HG00152")
    dataset = SingleSampleDataset(FASTA_PATH, TARGET_PATH, GTF_PATH)
    indices = list(range(len(dataset)))
    
    print("\n[TEST 1] MODEL A: ORIGINAL (GELU)")
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    # Testing on 200 samples
    t_dl = DataLoader(Subset(dataset, train_idx[:200]), batch_size=BATCH_SIZE, shuffle=True)
    v_dl = DataLoader(Subset(dataset, val_idx[:50]), batch_size=BATCH_SIZE)
    
    model_a = EnformerFineTuner(mode="Original").to(DEVICE)
    score_a = train_and_eval(model_a, t_dl, v_dl, EPOCHS)
    print(f"Result Model A: {score_a:.5f}")

    print("\n[TEST 2] MODEL B: MODIFIED (SwiGLU)")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_b = []
    
    for i, (t_idx, v_idx) in enumerate(kfold.split(indices)):
        if i > 0: break 
        print(f"   Running Fold {i+1}/5...")
        t_dl = DataLoader(Subset(dataset, t_idx[:200]), batch_size=BATCH_SIZE, shuffle=True)
        v_dl = DataLoader(Subset(dataset, v_idx[:50]), batch_size=BATCH_SIZE)
        model_b = EnformerFineTuner(mode="SwiGLU").to(DEVICE)
        scores_b.append(train_and_eval(model_b, t_dl, v_dl, EPOCHS))
        
    score_b = np.mean(scores_b)
    print(f"Result Model B: {score_b:.5f}")

    print("\n" + "="*30)
    print(f"Original: {score_a:.5f}")
    print(f"SwiGLU:   {score_b:.5f}")
    if score_b < score_a: print("WINNER: SwiGLU")
    else: print("WINNER: Original")

if __name__ == "__main__":
    main()
EOF

source /root/miniconda3/etc/profile.d/conda.sh
conda activate enformer_stable
nohup python experiment_comparison.py > experiment.log 2>&1 &
echo ">>> RESTARTED V6 (STABILIZED). Watch progress:"
echo "    tail -f experiment.log"

WORKDIR="/workspace/championship"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate enformer_stable

cat << 'EOF' > experiment_final_round.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from enformer_pytorch import Enformer
from sklearn.model_selection import KFold
import kipoiseq
from kipoiseq import Interval
import os

SEQ_LENGTH = 196_608 
BATCH_SIZE = 1 
LEARNING_RATE = 1e-4
EPOCHS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS = {
    "Original (A)": 0.30166,
    "SwiGLU (B)":   0.13774
}

FASTA_PATH = "/workspace/geuvadis/vcf/HG00152_chr22_personalized.fa"
TARGET_PATH = "/workspace/geuvadis/salmon_out/HG00152_quant/quant.sf"
GTF_PATH = "/workspace/geuvadis/reference/gencode.v44.annotation.gtf"

print(f">>> Running on: {DEVICE}")

class SingleSampleDataset(Dataset):
    def __init__(self, fasta_path, target_path, gtf_path):
        print(f"Loading data...")
        try:
            quant_df = pd.read_csv(target_path, sep="\t")
            self.tpm_map = dict(zip(quant_df['Name'], quant_df['TPM']))
        except: sys.exit(1)
            
        if not os.path.exists(fasta_path): sys.exit(1)
        self.extractor = kipoiseq.extractors.FastaStringExtractor(fasta_path)
        self.chrom_name = list(self.extractor.fasta.keys())[0]
        self.one_hot = kipoiseq.transforms.OneHot(alphabet="ACGT", neutral_value=0)
        self.base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        self.samples = []
        with open(gtf_path) as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split('\t')
                if parts[0] not in ['22', 'chr22']: continue
                if parts[2] == 'transcript':
                    try:
                        tid = [x for x in parts[8].split(';') if 'transcript_id' in x][0].split('"')[1]
                        if tid in self.tpm_map:
                            val = np.log1p(self.tpm_map[tid])
                            start, end = int(parts[3]), int(parts[4])
                            tss = start if parts[6] == '+' else end
                            self.samples.append({'tss': tss, 'target': val})
                    except: continue
        print(f"Dataset Ready: Found {len(self.samples)} transcripts.")

    def __len__(self): return len(self.samples)

    def robust_one_hot(self, seq):
        arr = np.zeros((len(seq), 4), dtype=np.float32)
        for i, char in enumerate(seq.upper()):
            if char in self.base_map:
                arr[i, self.base_map[char]] = 1.0
        return arr

    def __getitem__(self, idx):
        item = self.samples[idx]
        interval = Interval(self.chrom_name, item['tss'] - SEQ_LENGTH // 2, item['tss'] + SEQ_LENGTH // 2)
        try: seq = self.extractor.extract(interval)
        except: seq = "N" * SEQ_LENGTH
        if len(seq) < SEQ_LENGTH: seq += "N" * (SEQ_LENGTH - len(seq))
        elif len(seq) > SEQ_LENGTH: seq = seq[:SEQ_LENGTH]
        seq_enc = self.robust_one_hot(seq)
        return torch.tensor(seq_enc, dtype=torch.float32), torch.tensor([item['target']], dtype=torch.float32)

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class MHAPooling(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(input_dim, num_heads, bias=False) 
    def forward(self, x):
        attn_logits = self.query(x) 
        attn_weights = F.softmax(attn_logits, dim=1) 
        pooled_heads = []
        for h in range(self.num_heads):
            w = attn_weights[:, :, h].unsqueeze(-1) 
            pooled_h = (x * w).sum(dim=1) 
            pooled_heads.append(pooled_h)
        return torch.cat(pooled_heads, dim=-1)

class EnformerChallenger(nn.Module):
    def __init__(self, mode="MHA"):
        super().__init__()
        self.mode = mode
        self.backbone = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
        for param in self.backbone.parameters(): param.requires_grad = False
        
        dim = 5313 
        self.norm = nn.LayerNorm(dim)
        
        if mode == "MHA": 
            self.project = nn.Linear(dim, dim * 2)
            self.activation = SwiGLU() 
            self.pool_layer = MHAPooling(dim, num_heads=4)
            self.final = nn.Linear(dim * 4, 1) 
            
        elif mode == "ConvMish": 
            self.project = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),
                Mish(),
                nn.Dropout(0.1)
            )
            self.activation = nn.Identity() 
            self.pool_head = nn.Linear(dim, 1)
            self.final = nn.Linear(dim, 1)

    def forward(self, x):
        out = self.backbone(x)['human']
        mid = out.shape[1] // 2
        out = out[:, mid-2:mid+2, :] 
        out = self.norm(out)
        
        if self.mode == "ConvMish":
            out = out.permute(0, 2, 1) 
            out = self.project(out)
            out = out.permute(0, 2, 1) 
        else:
            out = self.activation(self.project(out))
        
        if self.mode == "MHA":
            pooled = self.pool_layer(out)
        else:
            if not hasattr(self, 'pool_head'):
                self.pool_head = nn.Linear(5313, 1).to(out.device)
            
            weights = F.softmax(self.pool_head(out), dim=1)
            pooled = (out * weights).sum(dim=1)
            
        return self.final(pooled)

def train_and_eval(model, train_loader, val_loader, epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    crit = nn.MSELoss()
    
    for ep in range(epochs):
        model.train()
        for seq, y in train_loader:
            seq, y = seq.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(seq)
            loss = crit(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            opt.step()
            
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for seq, y in val_loader:
                seq, y = seq.to(DEVICE), y.to(DEVICE)
                v_loss += crit(model(seq), y).item()
        
    return v_loss / len(val_loader)

def main():
    print(">>> INITIALIZING ROUND 2 (MODELS C & D)")
    dataset = SingleSampleDataset(FASTA_PATH, TARGET_PATH, GTF_PATH)
    indices = list(range(len(dataset)))
    subset_indices = indices[:250]
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    MODELS = ["MHA", "ConvMish"]

    for model_name in MODELS:
        print(f"\n>>> Training {model_name}...")
        scores = []
        
        for i, (t_idx, v_idx) in enumerate(kfold.split(subset_indices)):
            if i > 0: break # Single fold for speed
            
            t_dl = DataLoader(Subset(dataset, t_idx), batch_size=BATCH_SIZE, shuffle=True)
            v_dl = DataLoader(Subset(dataset, v_idx), batch_size=BATCH_SIZE)
            
            model = EnformerChallenger(mode=model_name).to(DEVICE)
            score = train_and_eval(model, t_dl, v_dl, EPOCHS)
            scores.append(score)
            
        RESULTS[model_name] = np.mean(scores)
        print(f"    Score: {RESULTS[model_name]:.5f}")

    print("\n" + "="*40)
    print("FINAL LEADERBOARD (MSE LOSS)")
    print("="*40)
    for name, score in sorted(RESULTS.items(), key=lambda item: item[1]):
        print(f"{name:15} : {score:.5f}")

if __name__ == "__main__":
    main()
EOF

echo ">>> Launching Final Round..."
pkill -f experiment_final_round.py || true
nohup python experiment_final_round.py > experiment_final.log 2>&1 &

echo ">>> RUNNING. Tracking logs:"
tail -f experiment_final.log
