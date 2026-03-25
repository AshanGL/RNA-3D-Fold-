"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          STANFORD RNA 3D FOLDING — FULL ADVANCED END-TO-END PIPELINE        ║
║  Architecture: MSA + Geometry + Structure + Fusion Transformer + 3D Fold    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Pipeline Stages:
  1. Data Loading & Preprocessing (MSA, PDB_RNA, CSV)
  2. Feature Engineering (embeddings, covariation, geometry, structure)
  3. Model Architecture (modular multi-branch transformer)
  4. Training Loop (with loss, scheduler, validation)
  5. Inference & Coordinate Prediction
  6. Post-processing & Submission
"""

# ─────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────
import os, math, gc, warnings, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from Bio.PDB import MMCIFParser
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True

# ─────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────
class Config:
    # Paths
    BASE       = "/kaggle/input/competitions/stanford-rna-3d-folding-2"
    MSA_DIR    = f"{BASE}/MSA"
    PDB_DIR    = f"{BASE}/PDB_RNA"
    TRAIN_CSV  = f"{BASE}/train_sequences.csv"
    VALID_CSV  = f"{BASE}/validation_sequences.csv"
    TEST_CSV   = f"{BASE}/test_sequences.csv"
    OUT_DIR    = "/kaggle/working"

    # Model
    D_MODEL    = 256          # main hidden dim
    D_PAIR     = 128          # pair representation dim
    N_HEADS    = 8
    N_LAYERS   = 6            # transformer layers per module
    DROPOUT    = 0.1
    MAX_LEN    = 512          # max sequence length (pad/crop)

    # MSA
    MAX_MSA_SEQS = 128        # max homologs to load

    # Training
    BATCH_SIZE = 4
    EPOCHS     = 30
    LR         = 3e-4
    WEIGHT_DECAY = 1e-2
    GRAD_CLIP  = 1.0
    WARMUP_STEPS = 500

    # Misc
    SEED       = 42
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

cfg = Config()

def seed_everything(seed=cfg.SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

seed_everything()
print(f"Device: {cfg.DEVICE}")

# ─────────────────────────────────────────────────────────────
# 2. CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────
VOCAB = {'A': 0, 'U': 1, 'G': 2, 'C': 3, '<PAD>': 4, '<UNK>': 5}
VOCAB_SIZE = len(VOCAB)

RNA_MAP = {
    'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C',
    'RA': 'A', 'RU': 'U', 'RG': 'G', 'RC': 'C',
    '2MG': 'G', '1MG': 'G', '7MG': 'G', '5MC': 'C',
    'H2U': 'U', 'PSU': 'U', 'I': 'G', 'M2G': 'G',
}

BACKBONE_ATOMS = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "N1", "N9"]

def encode_sequence(seq: str, max_len: int = cfg.MAX_LEN) -> torch.Tensor:
    """Encode RNA string → integer tensor, padded to max_len."""
    ids = [VOCAB.get(c, VOCAB['<UNK>']) for c in seq[:max_len]]
    ids += [VOCAB['<PAD>']] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

def seq_mask(seq: str, max_len: int = cfg.MAX_LEN) -> torch.Tensor:
    """Boolean mask: True = real token."""
    L = min(len(seq), max_len)
    mask = torch.zeros(max_len, dtype=torch.bool)
    mask[:L] = True
    return mask

# ─────────────────────────────────────────────────────────────
# 3. MSA PARSER & COVARIATION
# ─────────────────────────────────────────────────────────────
def parse_msa(target_id: str, max_seqs: int = cfg.MAX_MSA_SEQS) -> Optional[np.ndarray]:
    """
    Load MSA fasta → numpy array of shape (n_seqs, seq_len) with integer encoding.
    Returns None if file not found.
    """
    fasta_path = Path(cfg.MSA_DIR) / f"{target_id}.MSA.fasta"
    if not fasta_path.exists():
        # try case variants
        for p in Path(cfg.MSA_DIR).glob(f"{target_id.upper()}*.fasta"):
            fasta_path = p; break
        else:
            return None

    seqs = []
    current = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current:
                    seqs.append(''.join(current))
                current = []
            else:
                current.append(line)
        if current:
            seqs.append(''.join(current))

    if not seqs:
        return None

    query_len = len(seqs[0].replace('-', ''))

    encoded = []
    for s in seqs[:max_seqs]:
        row = []
        for c in s:
            row.append({'A':0,'U':1,'G':2,'C':3,'-':4}.get(c.upper(), 5))
        encoded.append(row[:query_len])

    # pad to uniform length
    max_col = max(len(r) for r in encoded)
    matrix = np.full((len(encoded), max_col), 4, dtype=np.int8)
    for i, row in enumerate(encoded):
        matrix[i, :len(row)] = row

    return matrix  # (n_seqs, L)


def compute_covariation_matrix(msa: np.ndarray, L: int) -> np.ndarray:
    """
    Compute mutual information (MI) covariation matrix from MSA.
    Returns (L, L) float32 matrix.
    """
    if msa is None or msa.shape[0] < 2:
        return np.zeros((L, L), dtype=np.float32)

    msa_clipped = msa[:, :L]  # (N, L)
    N, l = msa_clipped.shape
    if l < L:
        pad = np.full((N, L - l), 4, dtype=np.int8)
        msa_clipped = np.concatenate([msa_clipped, pad], axis=1)

    # compute single-column frequencies
    def col_freq(col):
        freqs = np.zeros(6)
        for v in col:
            freqs[min(v, 5)] += 1
        return freqs / (freqs.sum() + 1e-8)

    # mutual information between each pair of columns
    MI = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        fi = col_freq(msa_clipped[:, i])
        for j in range(i + 1, L):
            fj = col_freq(msa_clipped[:, j])
            # joint freq
            fij = np.zeros((6, 6))
            for k in range(N):
                fij[min(msa_clipped[k, i], 5), min(msa_clipped[k, j], 5)] += 1
            fij /= (N + 1e-8)
            # MI
            mi = 0.0
            for a in range(6):
                for b in range(6):
                    if fij[a, b] > 0 and fi[a] > 0 and fj[b] > 0:
                        mi += fij[a, b] * np.log(fij[a, b] / (fi[a] * fj[b] + 1e-8))
            MI[i, j] = MI[j, i] = mi

    # normalize to [0, 1]
    max_mi = MI.max()
    if max_mi > 0:
        MI /= max_mi
    return MI


# ─────────────────────────────────────────────────────────────
# 4. PDB / CIF COORDINATE PARSER
# ─────────────────────────────────────────────────────────────
_cif_parser = MMCIFParser(QUIET=True)

def load_coordinates(target_id: str) -> Optional[Tuple[str, np.ndarray]]:
    """
    Load CIF → (sequence_str, coords_array).
    coords: (L, 3) using C1' atom; NaN where missing.
    """
    cif_path = Path(cfg.PDB_DIR) / f"{target_id.lower()}.cif"
    if not cif_path.exists():
        return None

    structure = _cif_parser.get_structure(target_id, str(cif_path))
    model = structure[0]

    cif_seq, coords = [], []
    for chain in model:
        for residue in chain:
            resname = residue.resname.strip()
            base = RNA_MAP.get(resname, RNA_MAP.get(resname[-1] if resname else 'X', None))
            if base is None:
                continue
            cif_seq.append(base)
            if "C1'" in residue:
                coords.append(residue["C1'"].coord)
            else:
                coords.append([np.nan, np.nan, np.nan])

    if not cif_seq:
        return None
    return ''.join(cif_seq), np.array(coords, dtype=np.float32)


def align_and_extract(csv_seq: str, cif_seq: str,
                      coords: np.ndarray) -> Optional[np.ndarray]:
    """
    Align CSV sequence to CIF sequence, return coordinates for CSV portion.
    Returns (L_csv, 3) array or None.
    """
    L = len(csv_seq)
    # exact match
    idx = cif_seq.find(csv_seq)
    if idx != -1:
        return coords[idx:idx + L]

    # sliding window match (first 20 chars)
    for i in range(len(cif_seq) - 20):
        if cif_seq[i:i + 20] == csv_seq[:20]:
            end = min(i + L, len(cif_seq))
            c = coords[i:end]
            if len(c) < L:
                pad = np.full((L - len(c), 3), np.nan)
                c = np.concatenate([c, pad], axis=0)
            return c

    # fallback: return first L coords
    if len(coords) >= L:
        return coords[:L]
    pad = np.full((L - len(coords), 3), np.nan)
    return np.concatenate([coords, pad], axis=0)


# ─────────────────────────────────────────────────────────────
# 5. GEOMETRY FEATURES
# ─────────────────────────────────────────────────────────────
def compute_distance_matrix(coords: np.ndarray, max_len: int = cfg.MAX_LEN) -> np.ndarray:
    """
    coords: (L, 3). Returns (max_len, max_len) distance matrix (Angstrom).
    NaN positions filled with 0.
    """
    L = min(len(coords), max_len)
    D = np.zeros((max_len, max_len), dtype=np.float32)

    valid = ~np.isnan(coords[:L, 0])
    c = coords[:L].copy()
    c[~valid] = 0.0

    for i in range(L):
        for j in range(L):
            if valid[i] and valid[j]:
                d = np.linalg.norm(c[i] - c[j])
                D[i, j] = d

    # normalize
    max_d = D.max()
    if max_d > 0:
        D /= max_d
    return D


def compute_dihedral_features(coords: np.ndarray, max_len: int = cfg.MAX_LEN) -> np.ndarray:
    """
    Compute backbone torsion angles (pseudo-dihedral) for each residue.
    Returns (max_len, 2) sin/cos features.
    """
    L = min(len(coords), max_len)
    feats = np.zeros((max_len, 2), dtype=np.float32)

    for i in range(1, L - 2):
        try:
            p = [coords[i-1], coords[i], coords[i+1], coords[i+2]]
            if any(np.isnan(pi).any() for pi in p):
                continue

            b1 = p[1] - p[0]
            b2 = p[2] - p[1]
            b3 = p[3] - p[2]

            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)

            norm_n1 = np.linalg.norm(n1)
            norm_n2 = np.linalg.norm(n2)
            if norm_n1 < 1e-8 or norm_n2 < 1e-8:
                continue

            n1 /= norm_n1; n2 /= norm_n2

            cos_a = np.clip(np.dot(n1, n2), -1, 1)
            angle = np.arccos(cos_a)
            feats[i, 0] = np.sin(angle)
            feats[i, 1] = np.cos(angle)
        except:
            pass

    return feats


# ─────────────────────────────────────────────────────────────
# 6. SECONDARY STRUCTURE (SIMPLE NUSSINOV)
# ─────────────────────────────────────────────────────────────
PAIRS = {('A','U'), ('U','A'), ('G','C'), ('C','G'), ('G','U'), ('U','G')}

def nussinov_fold(seq: str, min_loop: int = 4) -> np.ndarray:
    """
    Classic Nussinov algorithm → pairing probability matrix (L, L).
    Returns float32 binary contact map.
    """
    L = len(seq)
    dp = np.zeros((L, L), dtype=np.int32)

    for span in range(min_loop + 1, L):
        for i in range(L - span):
            j = i + span
            # unpaired
            dp[i][j] = max(dp[i][j - 1], dp[i + 1][j] if i + 1 <= j else 0)
            # pair i-j
            if (seq[i], seq[j]) in PAIRS:
                inner = dp[i + 1][j - 1] if i + 1 <= j - 1 else 0
                dp[i][j] = max(dp[i][j], inner + 1)
            # bifurcation
            for k in range(i + 1, j):
                dp[i][j] = max(dp[i][j], dp[i][k] + dp[k + 1][j])

    # traceback
    contact = np.zeros((L, L), dtype=np.float32)

    def traceback(i, j):
        if i >= j:
            return
        if dp[i][j] == dp[i][j - 1]:
            traceback(i, j - 1)
        elif i + 1 <= j and dp[i][j] == dp[i + 1][j]:
            traceback(i + 1, j)
        elif (seq[i], seq[j]) in PAIRS:
            inner = dp[i + 1][j - 1] if i + 1 <= j - 1 else 0
            if dp[i][j] == inner + 1:
                contact[i][j] = contact[j][i] = 1.0
                traceback(i + 1, j - 1)
                return
        for k in range(i + 1, j):
            if dp[i][j] == dp[i][k] + dp[k + 1][j]:
                traceback(i, k)
                traceback(k + 1, j)
                return

    traceback(0, L - 1)
    return contact


def secondary_structure_features(seq: str, max_len: int = cfg.MAX_LEN) -> np.ndarray:
    """Contact map (max_len, max_len) from Nussinov."""
    L = min(len(seq), max_len)
    short = seq[:L]
    contact_short = nussinov_fold(short)
    contact = np.zeros((max_len, max_len), dtype=np.float32)
    contact[:L, :L] = contact_short
    return contact


# ─────────────────────────────────────────────────────────────
# 7. DATASET
# ─────────────────────────────────────────────────────────────
class RNADataset(Dataset):
    """
    Returns per-sample feature dict:
      - seq_ids   : (L,) token ids
      - seq_mask  : (L,) bool
      - msa_feat  : (L, L) covariation matrix
      - dist_feat : (L, L) distance matrix (or zeros for test)
      - dihed_feat: (L, 2) dihedral angles
      - contact   : (L, L) secondary structure contacts
      - coords    : (L, 3) ground truth coordinates (or zeros for test)
      - coord_mask: (L,) True where coords are valid
      - seq_len   : actual length (int)
    """

    def __init__(self, csv_path: str, is_test: bool = False):
        self.df = pd.read_csv(csv_path)
        self.is_test = is_test
        self.L = cfg.MAX_LEN

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target_id = row['target_id']
        seq = row['sequence']
        seq_len = min(len(seq), self.L)
        L = self.L

        # ── Token IDs & mask ──
        seq_ids = encode_sequence(seq, L)
        mask = seq_mask(seq, L)

        # ── MSA covariation ──
        msa = parse_msa(target_id)
        cov = compute_covariation_matrix(msa, seq_len)
        msa_feat = np.zeros((L, L), dtype=np.float32)
        msa_feat[:seq_len, :seq_len] = cov[:seq_len, :seq_len]

        # ── Secondary structure ──
        contact = secondary_structure_features(seq, L)

        # ── Coordinates & geometry features ──
        coords_gt  = np.zeros((L, 3), dtype=np.float32)
        coord_mask = np.zeros(L, dtype=np.float32)
        dist_feat  = np.zeros((L, L), dtype=np.float32)
        dihed_feat = np.zeros((L, 2), dtype=np.float32)

        if not self.is_test:
            result = load_coordinates(target_id)
            if result is not None:
                cif_seq, raw_coords = result
                aligned = align_and_extract(seq[:seq_len], cif_seq, raw_coords)
                if aligned is not None:
                    valid = ~np.isnan(aligned[:, 0])
                    coords_gt[:seq_len][valid] = aligned[:seq_len][valid]
                    coord_mask[:seq_len] = valid.astype(np.float32)
                    dist_feat  = compute_distance_matrix(aligned, L)
                    dihed_feat = compute_dihedral_features(aligned, L)

        return {
            'seq_ids'   : seq_ids,
            'seq_mask'  : mask,
            'msa_feat'  : torch.from_numpy(msa_feat),
            'dist_feat' : torch.from_numpy(dist_feat),
            'dihed_feat': torch.from_numpy(dihed_feat),
            'contact'   : torch.from_numpy(contact),
            'coords'    : torch.from_numpy(coords_gt),
            'coord_mask': torch.from_numpy(coord_mask),
            'seq_len'   : seq_len,
            'target_id' : target_id,
        }


def collate_fn(batch):
    keys_tensor = ['seq_ids', 'seq_mask', 'msa_feat', 'dist_feat',
                   'dihed_feat', 'contact', 'coords', 'coord_mask']
    out = {}
    for k in keys_tensor:
        out[k] = torch.stack([b[k] for b in batch])
    out['seq_len']   = [b['seq_len'] for b in batch]
    out['target_id'] = [b['target_id'] for b in batch]
    return out


# ─────────────────────────────────────────────────────────────
# 8. MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────

# ── 8a. Positional Encoding ──────────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = cfg.MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ── 8b. Pair Bias Attention ───────────────────────────────────
class PairBiasAttention(nn.Module):
    """
    Multi-head attention with additive pair bias.
    pair_bias: (B, L, L) → broadcast to (B, H, L, L).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.pair_proj = nn.Linear(1, n_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pair_bias, key_padding_mask=None):
        """
        x          : (B, L, D)
        pair_bias  : (B, L, L)
        """
        B, L, D = x.shape
        # (B, L, L, 1) → (B, L, L, H) → (B*H, L, L) for attn_mask
        bias = self.pair_proj(pair_bias.unsqueeze(-1))   # (B, L, L, H)
        bias = bias.permute(0, 3, 1, 2).reshape(B * self.attn.num_heads, L, L)

        # MultiheadAttention accepts attn_mask (L,L) or (B*H, L, L)
        out, _ = self.attn(x, x, x,
                           attn_mask=bias,
                           key_padding_mask=~key_padding_mask if key_padding_mask is not None else None)
        return self.dropout(out)


# ── 8c. Transformer Block ─────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn  = PairBiasAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, pair_bias, mask=None):
        x = x + self.attn(self.norm1(x), pair_bias, mask)
        x = x + self.ff(self.norm2(x))
        return x


# ── 8d. MSA MODULE ────────────────────────────────────────────
class MSAModule(nn.Module):
    """
    Takes covariation matrix → pair features.
    Input : (B, L, L)
    Output: (B, L, L, D_PAIR)
    """
    def __init__(self, d_pair: int = cfg.D_PAIR):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, d_pair),
        )

    def forward(self, cov):
        # cov: (B, L, L) → (B, L, L, 1)
        return self.proj(cov.unsqueeze(-1))  # (B, L, L, D_PAIR)


# ── 8e. GEOMETRY MODULE ───────────────────────────────────────
class GeometryModule(nn.Module):
    """
    Takes distance matrix + dihedral angles → node + pair features.
    dist  : (B, L, L)
    dihed : (B, L, 2)
    """
    def __init__(self, d_model: int = cfg.D_MODEL, d_pair: int = cfg.D_PAIR):
        super().__init__()
        # node branch: dihedrals
        self.node_proj = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, d_model),
        )
        # pair branch: distances
        self.pair_proj = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, d_pair),
        )

    def forward(self, dist, dihed):
        node_feat = self.node_proj(dihed)                   # (B, L, D)
        pair_feat = self.pair_proj(dist.unsqueeze(-1))      # (B, L, L, D_PAIR)
        return node_feat, pair_feat


# ── 8f. STRUCTURE MODULE ──────────────────────────────────────
class StructureModule(nn.Module):
    """
    Takes contact map (secondary structure) → pair features.
    contact: (B, L, L)
    """
    def __init__(self, d_pair: int = cfg.D_PAIR):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, d_pair),
        )

    def forward(self, contact):
        return self.proj(contact.unsqueeze(-1))  # (B, L, L, D_PAIR)


# ── 8g. FUSION MODULE ─────────────────────────────────────────
class FusionModule(nn.Module):
    """
    Fuses MSA + Geometry + Structure pair features → single pair bias.
    Also combines node features.
    """
    def __init__(self, d_model: int = cfg.D_MODEL, d_pair: int = cfg.D_PAIR,
                 n_heads: int = cfg.N_HEADS, n_layers: int = cfg.N_LAYERS):
        super().__init__()
        # fuse three pair tensors
        self.pair_fuse = nn.Sequential(
            nn.Linear(d_pair * 3, d_pair), nn.ReLU(),
            nn.Linear(d_pair, 1),
        )
        # single bias per position pair
        self.pair_to_bias = nn.Linear(d_pair, 1)

        # transformer on node features
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

    def forward(self, node_feat, msa_pair, geo_pair, struct_pair, mask=None):
        """
        node_feat : (B, L, D)
        *_pair    : (B, L, L, D_PAIR)
        Returns   : (B, L, D)
        """
        # fuse pair features → single (B, L, L) bias
        combined = torch.cat([msa_pair, geo_pair, struct_pair], dim=-1)  # (B,L,L,3*D_P)
        pair_bias = self.pair_fuse(combined).squeeze(-1)                 # (B, L, L)

        x = node_feat
        for layer in self.layers:
            x = layer(x, pair_bias, mask)
        return x, pair_bias


# ── 8h. 3D FOLDING MODULE ─────────────────────────────────────
class FoldingModule(nn.Module):
    """
    Projects fused node features → (x, y, z) coordinates.
    """
    def __init__(self, d_model: int = cfg.D_MODEL):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.head(x)  # (B, L, 3)


# ── 8i. FULL MODEL ────────────────────────────────────────────
class RNAFoldingModel(nn.Module):
    def __init__(self):
        super().__init__()
        D = cfg.D_MODEL
        DP = cfg.D_PAIR

        # Embedding
        self.embedding = nn.Embedding(VOCAB_SIZE, D, padding_idx=VOCAB['<PAD>'])
        self.pe         = SinusoidalPE(D)
        self.embed_norm = nn.LayerNorm(D)
        self.embed_drop = nn.Dropout(cfg.DROPOUT)

        # Modules
        self.msa_module    = MSAModule(DP)
        self.geo_module    = GeometryModule(D, DP)
        self.struct_module = StructureModule(DP)
        self.fusion        = FusionModule(D, DP, cfg.N_HEADS, cfg.N_LAYERS)
        self.folding       = FoldingModule(D)

        # Auxiliary heads (for multi-task loss)
        self.dist_head    = nn.Linear(1, 1)   # refine distance prediction
        self.contact_head = nn.Sequential(
            nn.Linear(D * 2, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch):
        seq_ids    = batch['seq_ids'].to(cfg.DEVICE)      # (B, L)
        seq_mask   = batch['seq_mask'].to(cfg.DEVICE)     # (B, L) bool
        msa_feat   = batch['msa_feat'].to(cfg.DEVICE)     # (B, L, L)
        dist_feat  = batch['dist_feat'].to(cfg.DEVICE)    # (B, L, L)
        dihed_feat = batch['dihed_feat'].to(cfg.DEVICE)   # (B, L, 2)
        contact    = batch['contact'].to(cfg.DEVICE)      # (B, L, L)

        # ── Embedding ──
        x = self.embedding(seq_ids)   # (B, L, D)
        x = self.pe(x)
        x = self.embed_drop(self.embed_norm(x))

        # ── Module branches ──
        msa_pair    = self.msa_module(msa_feat)                         # (B,L,L,DP)
        geo_node, geo_pair = self.geo_module(dist_feat, dihed_feat)     # (B,L,D), (B,L,L,DP)
        struct_pair = self.struct_module(contact)                       # (B,L,L,DP)

        # combine node features
        x = x + geo_node

        # ── Fusion ──
        fused, pair_bias = self.fusion(x, msa_pair, geo_pair, struct_pair, seq_mask)

        # ── 3D Coordinates ──
        coords = self.folding(fused)   # (B, L, 3)

        # ── Auxiliary: predicted contact ──
        B, L, D = fused.shape
        fi = fused.unsqueeze(2).expand(-1, -1, L, -1)   # (B, L, L, D)
        fj = fused.unsqueeze(1).expand(-1, L, -1, -1)   # (B, L, L, D)
        pred_contact = self.contact_head(torch.cat([fi, fj], dim=-1)).squeeze(-1)  # (B,L,L)

        return {
            'coords'      : coords,
            'pair_bias'   : pair_bias,
            'pred_contact': pred_contact,
        }


# ─────────────────────────────────────────────────────────────
# 9. LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────
def tm_score_loss(pred_coords, true_coords, mask, seq_len):
    """
    Differentiable TM-score-inspired loss on C1' coordinates.
    Lower = better (we minimize negative TM-like score).
    """
    eps = 1e-8
    B = pred_coords.shape[0]
    total = torch.zeros(1, device=pred_coords.device)

    for b in range(B):
        L = seq_len[b]
        d0 = 1.24 * (L - 15) ** (1/3) - 1.8 if L > 21 else 0.5
        d0 = max(d0, 0.5)

        m  = mask[b, :L].bool()
        if m.sum() < 2:
            continue

        p = pred_coords[b, :L][m]
        t = true_coords[b, :L][m]

        # center
        p = p - p.mean(0)
        t = t - t.mean(0)

        dist_sq = ((p - t) ** 2).sum(-1)
        tm_terms = 1.0 / (1.0 + dist_sq / (d0 ** 2))
        score = tm_terms.mean() / L * m.sum()
        total = total - score   # negate: we minimize

    return total / B


def coordinate_loss(pred, true, mask):
    """MSE loss only on valid (non-NaN) positions."""
    m = mask.bool().unsqueeze(-1).expand_as(pred)
    if m.sum() == 0:
        return pred.new_zeros(1).squeeze()
    return F.mse_loss(pred[m], true.to(pred.device)[m])


def contact_loss(pred_contact, true_contact, mask):
    """BCE on contact map, masked."""
    B, L, _ = pred_contact.shape
    m = mask.unsqueeze(2) * mask.unsqueeze(1)  # (B, L, L)
    if m.sum() == 0:
        return pred_contact.new_zeros(1).squeeze()
    return F.binary_cross_entropy(pred_contact * m, true_contact.to(pred_contact.device) * m)


def combined_loss(outputs, batch, seq_len):
    pred_coords   = outputs['coords']
    pred_contact  = outputs['pred_contact']

    true_coords   = batch['coords'].to(cfg.DEVICE)
    true_contact  = batch['contact'].to(cfg.DEVICE)
    coord_mask    = batch['coord_mask'].to(cfg.DEVICE)

    l_coord   = coordinate_loss(pred_coords, true_coords, coord_mask)
    l_tm      = tm_score_loss(pred_coords, true_coords, coord_mask, seq_len)
    l_contact = contact_loss(pred_contact, true_contact, coord_mask)

    total = l_coord + 0.5 * l_tm + 0.2 * l_contact
    return total, {'coord': l_coord.item(), 'tm': l_tm.item(), 'contact': l_contact.item()}


# ─────────────────────────────────────────────────────────────
# 10. TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    total_loss = 0.0
    loss_parts = {'coord': 0.0, 'tm': 0.0, 'contact': 0.0}
    n = 0

    bar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for step, batch in enumerate(bar):
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(cfg.DEVICE == 'cuda')):
            outputs = model(batch)
            loss, parts = combined_loss(outputs, batch, batch['seq_len'])

        if cfg.DEVICE == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item()
        for k in loss_parts:
            loss_parts[k] += parts[k]
        n += 1

        bar.set_postfix(loss=f"{total_loss/n:.4f}")

    return total_loss / n, {k: v / n for k, v in loss_parts.items()}


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss = 0.0; n = 0

    for batch in tqdm(loader, desc="Validation", leave=False):
        with torch.cuda.amp.autocast(enabled=(cfg.DEVICE == 'cuda')):
            outputs = model(batch)
            loss, _ = combined_loss(outputs, batch, batch['seq_len'])
        total_loss += loss.item(); n += 1

    return total_loss / n


def run_training():
    print("\n=== LOADING DATASETS ===")
    train_ds = RNADataset(cfg.TRAIN_CSV, is_test=False)
    valid_ds = RNADataset(cfg.VALID_CSV, is_test=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS)

    print(f"Train: {len(train_ds)} | Valid: {len(valid_ds)}")

    model = RNAFoldingModel().to(cfg.DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR * 0.01)
    scaler    = torch.cuda.amp.GradScaler(enabled=(cfg.DEVICE == 'cuda'))

    best_val = float('inf')
    history  = []

    print("\n=== TRAINING ===")
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, parts = train_epoch(model, train_loader, optimizer, scaler, epoch)
        val_loss          = eval_epoch(model, valid_loader)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{cfg.EPOCHS} | "
              f"Train {train_loss:.4f} (coord={parts['coord']:.3f} tm={parts['tm']:.3f} "
              f"contact={parts['contact']:.3f}) | "
              f"Val {val_loss:.4f} | LR {lr_now:.2e}")

        history.append({'epoch': epoch, 'train': train_loss, 'val': val_loss})

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(cfg.OUT_DIR, 'best_model.pt')
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_loss': val_loss}, ckpt)
            print(f"   ✅  Saved best model (val={best_val:.4f})")

        gc.collect()
        if cfg.DEVICE == 'cuda':
            torch.cuda.empty_cache()

    return model, history


# ─────────────────────────────────────────────────────────────
# 11. INFERENCE & SUBMISSION
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, loader) -> pd.DataFrame:
    """Run inference and return DataFrame with columns:
       target_id, resid, x_1, y_1, z_1  (using C1' as representative)
    """
    model.eval()
    records = []

    for batch in tqdm(loader, desc="Inference"):
        outputs = model(batch)
        pred_coords = outputs['coords'].cpu().numpy()  # (B, L, 3)

        for b, tid in enumerate(batch['target_id']):
            L = batch['seq_len'][b]
            coords = pred_coords[b, :L]
            for i, (x, y, z) in enumerate(coords):
                records.append({
                    'target_id': tid,
                    'resid'    : i + 1,
                    'x_1'      : float(x),
                    'y_1'      : float(y),
                    'z_1'      : float(z),
                })

    return pd.DataFrame(records)


def run_inference(model=None):
    if model is None:
        ckpt = os.path.join(cfg.OUT_DIR, 'best_model.pt')
        model = RNAFoldingModel().to(cfg.DEVICE)
        state = torch.load(ckpt, map_location=cfg.DEVICE)
        model.load_state_dict(state['model'])
        print(f"Loaded checkpoint (epoch {state['epoch']}, val={state['val_loss']:.4f})")

    # run on validation (scoring) and test
    for split, csv_path in [('validation', cfg.VALID_CSV), ('test', cfg.TEST_CSV)]:
        if not Path(csv_path).exists():
            print(f"[SKIP] {csv_path} not found")
            continue
        ds = RNADataset(csv_path, is_test=(split == 'test'))
        loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS)
        df_pred = predict(model, loader)
        out_path = os.path.join(cfg.OUT_DIR, f'predictions_{split}.csv')
        df_pred.to_csv(out_path, index=False)
        print(f"Saved {split} predictions → {out_path}  ({len(df_pred)} rows)")

    return model


# ─────────────────────────────────────────────────────────────
# 12. MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("RNA 3D FOLDING — ADVANCED END-TO-END PIPELINE")
    print("=" * 70)

    # Verify paths
    for name, path in [("TRAIN CSV", cfg.TRAIN_CSV), ("VALID CSV", cfg.VALID_CSV),
                       ("MSA DIR",   cfg.MSA_DIR),   ("PDB DIR",  cfg.PDB_DIR)]:
        status = "✅" if Path(path).exists() else "❌"
        print(f"  {status}  {name}: {path}")

    # Train
    model, history = run_training()

    # Inference
    run_inference(model)

    print("\n🎉 Pipeline complete!")
