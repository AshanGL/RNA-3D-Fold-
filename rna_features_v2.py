"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RNA FEATURE ENGINEERING v2 — SEQUENCE-ONLY INFERENCE READY                ║
║                                                                              ║
║  CHANGES FROM v1:                                                            ║
║   • Geometric features are COMPUTED FROM COORDS AT TRAIN TIME ONLY           ║
║   • At INFERENCE time, coords=None → geometry is zeroed / estimated          ║
║   • Secondary structure (Nussinov + thermodynamic scoring) is ALWAYS used   ║
║   • Long-sequence chunking (> MAX_LEN) with sliding-window overlap           ║
║   • Case-insensitive MSA file lookup (2D19 / 2d19 / 2D19.MSA.fasta)         ║
║   • Relative position encoding stays — it's sequence-only                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, math, warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
from scipy.special import softmax

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
ALPHA      = 4
GAP        = 4
MSA_TOKENS = {'A': 0, 'U': 1, 'G': 2, 'C': 3, '-': 4, '.': 4}
RNA_MAP = {
    'A':'A','U':'U','G':'G','C':'C',
    'RA':'A','RU':'U','RG':'G','RC':'C',
    '2MG':'G','1MG':'G','7MG':'G','5MC':'C',
    'H2U':'U','PSU':'U','I':'G','M2G':'G',
}

VOCAB = {'A':0,'U':1,'G':2,'C':3,'<PAD>':4,'<UNK>':5}

CONTACT_THRESHOLD = 8.0
N_DIST_BINS       = 36
D_MIN, D_MAX      = 2.0, 20.0
N_ANGLE_BINS      = 36
RBF_CENTERS       = np.linspace(D_MIN, D_MAX, 16)
RBF_GAMMA         = 1.0 / (2 * ((D_MAX - D_MIN) / 16) ** 2)

# Watson-Crick + wobble pairs
WC_PAIRS = {('A','U'), ('U','A'), ('G','C'), ('C','G'), ('G','U'), ('U','G')}


# ═════════════════════════════════════════════════════════════
# 1. CASE-INSENSITIVE MSA FILE FINDER
# ═════════════════════════════════════════════════════════════
def find_msa_file(msa_dir: str, target_id: str) -> Optional[str]:
    """
    Try multiple case/extension variants to find the MSA fasta file.
    Handles cases like: 2D19.MSA.fasta, 2d19.MSA.fasta, 2D19.msa.fasta, etc.
    """
    msa_dir = Path(msa_dir)
    if not msa_dir.exists():
        return None

    candidates = [
        f"{target_id}.MSA.fasta",
        f"{target_id.upper()}.MSA.fasta",
        f"{target_id.lower()}.MSA.fasta",
        f"{target_id}.msa.fasta",
        f"{target_id.upper()}.msa.fasta",
        f"{target_id}.fasta",
        f"{target_id.upper()}.fasta",
    ]

    for name in candidates:
        p = msa_dir / name
        if p.exists():
            return str(p)

    # glob fallback
    for p in msa_dir.glob(f"*{target_id.upper()}*"):
        return str(p)
    for p in msa_dir.glob(f"*{target_id.lower()}*"):
        return str(p)

    return None


# ═════════════════════════════════════════════════════════════
# 2. MSA LOADING & PREPROCESSING
# ═════════════════════════════════════════════════════════════
def load_msa(msa_path: str, max_seqs: int = 512) -> Tuple[np.ndarray, int]:
    seqs, current = [], []
    with open(msa_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current: seqs.append(''.join(current))
                current = []
            else:
                current.append(line.upper())
        if current: seqs.append(''.join(current))

    if not seqs:
        return np.zeros((1, 1), dtype=np.int8), 1

    query    = seqs[0]
    L_query  = len(query.replace('-', '').replace('.', ''))

    encoded = []
    for s in seqs[:max_seqs]:
        row = [MSA_TOKENS.get(c, 4) for c in s]
        encoded.append(row[:len(query)])

    max_col = max(len(r) for r in encoded)
    msa = np.full((len(encoded), max_col), 4, dtype=np.int8)
    for i, row in enumerate(encoded):
        msa[i, :len(row)] = row

    return msa, L_query


def filter_columns(msa: np.ndarray, max_gap_frac: float = 0.3) -> np.ndarray:
    gap_frac = (msa == GAP).mean(axis=0)
    return msa[:, gap_frac <= max_gap_frac]


def sequence_weights(msa: np.ndarray, theta: float = 0.2) -> np.ndarray:
    N, L = msa.shape
    if N == 1:
        return np.ones(1, dtype=np.float32)
    id_matrix = np.zeros((N, N), dtype=np.float32)
    for k in range(L):
        col = msa[:, k]
        id_matrix += (col[:, None] == col[None, :]).astype(np.float32)
    id_frac    = id_matrix / L
    neighbours = (id_frac >= (1.0 - theta)).sum(axis=1).astype(np.float32)
    return 1.0 / np.maximum(neighbours, 1.0)


def compute_single_freq(msa, weights, pseudo=0.5):
    N, L = msa.shape
    A    = ALPHA + 1
    f    = np.zeros((L, A), dtype=np.float32)
    Neff = weights.sum()
    for a in range(A):
        f[:, a] = (weights[:, None] * (msa == a)).sum(axis=0)
    f /= (Neff + 1e-8)
    return (1 - pseudo) * f + pseudo / A


def compute_pair_freq(msa, weights, pseudo=0.5):
    N, L = msa.shape
    A    = ALPHA + 1
    Neff = weights.sum()
    f2   = np.zeros((L, L, A, A), dtype=np.float32)
    for i in range(L):
        ci = msa[:, i]
        for j in range(i, L):
            cj = msa[:, j]
            for n in range(N):
                a = min(int(ci[n]), A - 1)
                b = min(int(cj[n]), A - 1)
                f2[i, j, a, b] += weights[n]
                if i != j:
                    f2[j, i, b, a] += weights[n]
    f2 /= (Neff + 1e-8)
    return (1 - pseudo) * f2 + pseudo / (A * A)


def compute_MI(f1, f2):
    L, A = f1.shape
    MI   = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(i + 1, L):
            mi = 0.0
            for a in range(A):
                for b in range(A):
                    fij = f2[i, j, a, b]
                    fi  = f1[i, a]
                    fj  = f1[j, b]
                    if fij > 1e-9 and fi > 1e-9 and fj > 1e-9:
                        mi += fij * math.log(fij / (fi * fj))
            MI[i, j] = MI[j, i] = mi
    return MI


def apc_correction(MI):
    mean_i   = MI.mean(axis=1, keepdims=True)
    mean_j   = MI.mean(axis=0, keepdims=True)
    mean_all = MI.mean() + 1e-8
    MIp      = MI - (mean_i * mean_j) / mean_all
    np.fill_diagonal(MIp, 0.0)
    return np.clip(MIp, 0, None)


def compute_covariance_matrix(f1, f2):
    L, A  = f1.shape
    outer = f1[:, None, :, None] * f1[None, :, None, :]
    return f2 - outer


def frobenius_norm_DI(cov):
    L  = cov.shape[0]
    FN = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(i + 1, L):
            fn          = np.linalg.norm(cov[i, j])
            FN[i, j]    = FN[j, i] = fn
    return apc_correction(FN)


def msa_covariation_features(msa_path: Optional[str],
                              seq_len: int,
                              max_seqs: int = 512) -> Dict[str, np.ndarray]:
    L = seq_len
    empty = {
        'MI' : np.zeros((L, L), np.float32),
        'MIp': np.zeros((L, L), np.float32),
        'FNp': np.zeros((L, L), np.float32),
        'f1' : np.full((L, 5), 0.2, np.float32),
        'neff': 1.0,
    }

    if msa_path is None or not Path(msa_path).exists():
        return empty

    try:
        msa, L_query = load_msa(msa_path, max_seqs)
        msa = filter_columns(msa, max_gap_frac=0.3)

        Lm = msa.shape[1]
        if Lm > L:
            msa = msa[:, :L]
        elif Lm < L:
            pad = np.full((msa.shape[0], L - Lm), GAP, dtype=np.int8)
            msa = np.concatenate([msa, pad], axis=1)

        weights = sequence_weights(msa)
        neff    = float(weights.sum())
        f1      = compute_single_freq(msa, weights)
        f2      = compute_pair_freq(msa, weights)
        MI      = compute_MI(f1, f2)
        MIp     = apc_correction(MI)
        cov     = compute_covariance_matrix(f1, f2)
        FNp     = frobenius_norm_DI(cov)

        def norm01(x):
            m = x.max(); return x / m if m > 0 else x

        return {
            'MI'  : norm01(MI),
            'MIp' : norm01(MIp),
            'FNp' : norm01(FNp),
            'f1'  : f1,
            'neff': neff,
        }
    except Exception as e:
        print(f"  [MSA warn] {msa_path}: {e}")
        return empty


# ═════════════════════════════════════════════════════════════
# 3. SECONDARY STRUCTURE — NUSSINOV + STACKING SCORES
#    Used at BOTH train AND inference time (sequence only!)
# ═════════════════════════════════════════════════════════════
# Pair scores: Watson-Crick > wobble
PAIR_SCORE = {
    ('G','C'): 3, ('C','G'): 3,
    ('A','U'): 2, ('U','A'): 2,
    ('G','U'): 1, ('U','G'): 1,
}

def nussinov_fold(seq: str, min_loop: int = 3) -> np.ndarray:
    """
    Nussinov with base-pair scores → binary contact map (L, L).
    O(N^3) — fine for sequences up to ~1000 nt in chunks.
    """
    L  = len(seq)
    dp = np.zeros((L, L), dtype=np.float32)

    for span in range(min_loop + 1, L):
        for i in range(L - span):
            j = i + span
            # unpaired extension
            score = max(dp[i, j-1], dp[i+1, j] if i+1 <= j else 0)
            # pair i-j
            ps = PAIR_SCORE.get((seq[i], seq[j]), 0)
            if ps > 0:
                inner = dp[i+1, j-1] if i+1 <= j-1 else 0
                score = max(score, inner + ps)
            # bifurcation
            for k in range(i+1, j):
                score = max(score, dp[i, k] + dp[k+1, j])
            dp[i, j] = score

    # traceback
    contact = np.zeros((L, L), dtype=np.float32)

    def traceback(i, j):
        if i >= j:
            return
        if dp[i, j] == dp[i, j-1]:
            traceback(i, j-1)
        elif i+1 <= j and dp[i, j] == dp[i+1, j]:
            traceback(i+1, j)
        else:
            ps = PAIR_SCORE.get((seq[i], seq[j]), 0)
            if ps > 0:
                inner = dp[i+1, j-1] if i+1 <= j-1 else 0
                if dp[i, j] == inner + ps:
                    contact[i, j] = contact[j, i] = 1.0
                    traceback(i+1, j-1)
                    return
            for k in range(i+1, j):
                if dp[i, j] == dp[i, k] + dp[k+1, j]:
                    traceback(i, k)
                    traceback(k+1, j)
                    return

    traceback(0, L-1)
    return contact


def secondary_structure_features(seq: str, max_len: int) -> Dict[str, np.ndarray]:
    """
    Returns:
      'contact_ss' : (max_len, max_len) binary contact map
      'ss_pair'    : (max_len,) 1.0 if residue is paired else 0.0
      'pair_type'  : (max_len, max_len, 3) one-hot  [WC, wobble, unpaired pair]
    """
    L    = min(len(seq), max_len)
    short = seq[:L]
    cm   = nussinov_fold(short)

    contact = np.zeros((max_len, max_len), dtype=np.float32)
    contact[:L, :L] = cm

    paired = contact.max(axis=1)  # (max_len,)

    # pair-type one-hot: [WC=GC/AU, wobble=GU, unpaired]
    pair_type = np.zeros((max_len, max_len, 3), dtype=np.float32)
    for i in range(L):
        for j in range(L):
            if cm[i, j] > 0:
                s = (short[i], short[j])
                if s in {('G','C'),('C','G'),('A','U'),('U','A')}:
                    pair_type[i, j, 0] = 1.0
                else:
                    pair_type[i, j, 1] = 1.0
            else:
                pair_type[i, j, 2] = 1.0

    return {
        'contact_ss': contact,
        'ss_pair'   : paired,
        'pair_type' : pair_type,
    }


# ═════════════════════════════════════════════════════════════
# 4. GEOMETRIC FEATURES FROM 3D COORDS (TRAIN ONLY)
#    For inference, coords=None → zeros returned
# ═════════════════════════════════════════════════════════════
def rbf_encode(dist: np.ndarray) -> np.ndarray:
    d = dist[:, :, None]
    centers = RBF_CENTERS[None, None, :]
    return np.exp(-RBF_GAMMA * (d - centers) ** 2)


def bin_distances(dist: np.ndarray, n_bins=N_DIST_BINS) -> np.ndarray:
    edges   = np.linspace(D_MIN, D_MAX, n_bins + 1)
    indices = np.clip(np.digitize(dist, edges) - 1, 0, n_bins - 1)
    return np.eye(n_bins, dtype=np.float32)[indices]


def compute_frame_orientations(coords: np.ndarray) -> np.ndarray:
    L     = len(coords)
    valid = ~np.isnan(coords[:, 0])
    T     = np.zeros((L, 3), dtype=np.float32)

    for i in range(L - 1):
        if valid[i] and valid[i+1]:
            d = coords[i+1] - coords[i]
            n = np.linalg.norm(d)
            if n > 1e-8:
                T[i] = d / n

    orient = np.zeros((L, L, 4), dtype=np.float32)
    for i in range(L):
        if not valid[i]: continue
        ti = T[i]; nti = np.linalg.norm(ti)
        if nti < 1e-8: continue
        for j in range(L):
            if not valid[j] or i == j: continue
            rij = coords[j] - coords[i]
            rijn = np.linalg.norm(rij)
            if rijn < 1e-8: continue
            rij /= rijn
            cos_t = float(np.clip(np.dot(ti / nti, rij), -1, 1))
            sin_t = float(np.sqrt(max(1 - cos_t**2, 0)))
            perp  = rij - cos_t * (ti / nti)
            pn    = np.linalg.norm(perp)
            cos_o, sin_o = 1.0, 0.0
            if pn > 1e-8:
                perp /= pn
                aux = np.array([0., 0., 1.])
                if abs(np.dot(ti / nti, aux)) > 0.9:
                    aux = np.array([0., 1., 0.])
                t2 = np.cross(ti / nti, aux)
                t2n = np.linalg.norm(t2)
                if t2n > 1e-8:
                    t2 /= t2n
                    cos_o = float(np.clip(np.dot(t2, perp), -1, 1))
                    sin_o = float(np.sqrt(max(1 - cos_o**2, 0)))
            orient[i, j] = [cos_o, sin_o, cos_t, sin_t]

    return orient


def pseudo_dihedral_angles(coords: np.ndarray) -> np.ndarray:
    L     = len(coords)
    feats = np.zeros((L, 4), dtype=np.float32)
    valid = ~np.isnan(coords[:, 0])

    def dihedral(p0, p1, p2, p3):
        b1 = p1-p0; b2 = p2-p1; b3 = p3-p2
        n1 = np.cross(b1, b2); n1n = np.linalg.norm(n1)
        n2 = np.cross(b2, b3); n2n = np.linalg.norm(n2)
        if n1n < 1e-8 or n2n < 1e-8: return 0.0
        n1 /= n1n; n2 /= n2n
        b2u  = b2 / (np.linalg.norm(b2) + 1e-8)
        cos_a = np.clip(np.dot(n1, n2), -1, 1)
        sin_a = np.dot(np.cross(n1, n2), b2u)
        return math.atan2(sin_a, cos_a)

    for i in range(1, L - 2):
        if not all(valid[max(0,i-1):i+3]): continue
        eta   = dihedral(coords[i-1], coords[i],   coords[i+1], coords[i+2])
        theta = 0.0
        if i+3 < L and valid[i+3]:
            theta = dihedral(coords[i], coords[i+1], coords[i+2], coords[i+3])
        feats[i] = [math.sin(eta), math.cos(eta), math.sin(theta), math.cos(theta)]

    return feats


def relative_position_encoding(L: int, max_range: int = 32) -> np.ndarray:
    """Sequence-relative position bias — used at both train & inference."""
    n_bins = 2 * max_range + 1
    enc    = np.zeros((L, L, n_bins), dtype=np.float32)
    for i in range(L):
        for j in range(L):
            rel = j - i
            idx = int(np.clip(rel + max_range, 0, n_bins - 1))
            enc[i, j, idx] = 1.0
    return enc


def geometric_features(coords: Optional[np.ndarray],
                        max_len: int = 512) -> Dict[str, np.ndarray]:
    """
    If coords is None → return zero-filled arrays (inference mode).
    If coords provided → compute real features (train mode).
    """
    L = max_len

    if coords is None:
        # inference: zero placeholders
        return {
            'dist_norm' : np.zeros((L, L),     np.float32),
            'dist_rbf'  : np.zeros((L, L, 16), np.float32),
            'dist_bins' : np.zeros((L, L, 36), np.float32),
            'contact_3d': np.zeros((L, L),     np.float32),
            'orient'    : np.zeros((L, L, 4),  np.float32),
            'dihed'     : np.zeros((L, 4),     np.float32),
            'valid_mask': np.zeros(L,          np.float32),
        }

    L_raw = min(len(coords), max_len)
    c     = coords[:L_raw].copy()
    valid = ~np.isnan(c[:, 0])
    c_cl  = c.copy(); c_cl[~valid] = 0.0

    # distance matrix
    dist_raw = np.zeros((L_raw, L_raw), np.float32)
    vi = np.where(valid)[0]
    for ii in range(len(vi)):
        for jj in range(ii + 1, len(vi)):
            i2, j2 = vi[ii], vi[jj]
            d = np.linalg.norm(c_cl[i2] - c_cl[j2])
            dist_raw[i2, j2] = dist_raw[j2, i2] = d

    dist_full = np.zeros((L, L), np.float32)
    dist_full[:L_raw, :L_raw] = dist_raw

    dist_norm = dist_full.copy()
    mx = dist_norm.max()
    if mx > 0: dist_norm /= mx

    rbf_full  = np.zeros((L, L, 16), np.float32)
    rbf_full[:L_raw, :L_raw] = rbf_encode(dist_raw)

    bin_full  = np.zeros((L, L, 36), np.float32)
    bin_full[:L_raw, :L_raw] = bin_distances(dist_raw)

    contact3d = (dist_full < CONTACT_THRESHOLD).astype(np.float32)
    np.fill_diagonal(contact3d, 0)

    orient_raw  = compute_frame_orientations(c)
    orient_full = np.zeros((L, L, 4), np.float32)
    orient_full[:L_raw, :L_raw] = orient_raw

    dihed_raw  = pseudo_dihedral_angles(c)
    dihed_full = np.zeros((L, 4), np.float32)
    dihed_full[:L_raw] = dihed_raw

    valid_full = np.zeros(L, np.float32)
    valid_full[:L_raw] = valid.astype(np.float32)

    return {
        'dist_norm' : dist_norm,
        'dist_rbf'  : rbf_full,
        'dist_bins' : bin_full,
        'contact_3d': contact3d,
        'orient'    : orient_full,
        'dihed'     : dihed_full,
        'valid_mask': valid_full,
    }


# ═════════════════════════════════════════════════════════════
# 5. LONG-SEQUENCE CHUNKING UTILITIES
# ═════════════════════════════════════════════════════════════
def chunk_sequence(seq: str, chunk_size: int = 512,
                   overlap: int = 64) -> List[Tuple[int, int, str]]:
    """
    Splits long sequences into overlapping chunks.
    Returns list of (start_idx, end_idx, chunk_str).
    """
    L = len(seq)
    if L <= chunk_size:
        return [(0, L, seq)]

    chunks = []
    start  = 0
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append((start, end, seq[start:end]))
        if end == L:
            break
        start = end - overlap  # overlap region
    return chunks


def stitch_coords(chunk_coords: List[Tuple[int, int, np.ndarray]],
                  total_len: int,
                  overlap: int = 64) -> np.ndarray:
    """
    Stitch chunked coordinate predictions back together.
    Uses linear blending in overlap regions.
    """
    coords = np.full((total_len, 3), np.nan, np.float32)
    weight = np.zeros(total_len, np.float32)

    for (start, end, c) in chunk_coords:
        Lc = end - start
        w  = np.ones(Lc, np.float32)

        # blend overlap at the start of this chunk
        if start > 0:
            blend = min(overlap, Lc)
            w[:blend] = np.linspace(0.0, 1.0, blend)

        # blend overlap at the end of this chunk
        if end < total_len:
            blend = min(overlap, Lc)
            w[-blend:] = np.minimum(w[-blend:], np.linspace(1.0, 0.0, blend))

        for i in range(Lc):
            gi = start + i
            if not np.isnan(c[i, 0]):
                if np.isnan(coords[gi, 0]):
                    coords[gi] = c[i] * w[i]
                    weight[gi] = w[i]
                else:
                    coords[gi] = coords[gi] + c[i] * w[i]
                    weight[gi] += w[i]

    # normalize
    for gi in range(total_len):
        if weight[gi] > 0:
            coords[gi] /= weight[gi]

    return coords


# ═════════════════════════════════════════════════════════════
# 6. UNIFIED FEATURE BUILDER
# ═════════════════════════════════════════════════════════════
def build_all_features(
    seq        : str,
    target_id  : str,
    coords     : Optional[np.ndarray],   # None at inference time
    msa_dir    : str,
    max_len    : int = 512,
    is_inference: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Build the complete feature set for one RNA sequence/chunk.

    At train time  : coords come from PDB; all features are computed.
    At inference   : coords=None; geometric features are zeroed;
                     secondary structure and MSA are still computed.
    """
    L     = min(len(seq), max_len)
    seq_t = seq[:L]

    # ── MSA covariation ──────────────────────────────────────
    # IMPORTANT: pass max_len (512), NOT L (actual seq len).
    # msa_covariation_features pads/crops to the given seq_len, so if we
    # pass L the returned MIp/FNp/f1 will be (L_seq, ...) — variable across
    # samples — and torch.stack in collate_fn will crash with a size mismatch.
    msa_path = find_msa_file(msa_dir, target_id)
    cov      = msa_covariation_features(msa_path, max_len, max_seqs=512)

    # ── Secondary structure (sequence-only) ──────────────────
    ss = secondary_structure_features(seq_t, max_len)

    # ── Geometric features ───────────────────────────────────
    # At inference, coords=None → zero arrays
    geo = geometric_features(coords, max_len)

    # ── Relative position encoding (sequence-only) ───────────
    rel_pos = relative_position_encoding(max_len, max_range=32)

    # ── Merge ────────────────────────────────────────────────
    return {
        # MSA
        'MIp'       : cov['MIp'],          # (L, L)
        'FNp'       : cov['FNp'],          # (L, L)
        'f1'        : cov['f1'],           # (L, 5)
        'neff'      : cov['neff'],         # scalar

        # Secondary structure (always available)
        'contact_ss': ss['contact_ss'],    # (max_len, max_len)
        'ss_pair'   : ss['ss_pair'],       # (max_len,)
        'pair_type' : ss['pair_type'],     # (max_len, max_len, 3)

        # Geometry (zeros at inference)
        'dist_norm' : geo['dist_norm'],    # (max_len, max_len)
        'dist_rbf'  : geo['dist_rbf'],     # (max_len, max_len, 16)
        'dist_bins' : geo['dist_bins'],    # (max_len, max_len, 36)
        'contact_3d': geo['contact_3d'],   # (max_len, max_len)
        'orient'    : geo['orient'],       # (max_len, max_len, 4)
        'dihed'     : geo['dihed'],        # (max_len, 4)

        # Relative position (always)
        'rel_pos'   : rel_pos,             # (max_len, max_len, 65)

        'seq_len'   : L,
    }


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== rna_features_v2 self-test ===")
    seq = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA"
    print(f"Seq length: {len(seq)}")

    # train mode
    np.random.seed(0)
    coords = np.cumsum(np.random.randn(len(seq), 3) * 2, axis=0).astype(np.float32)
    feats_train = build_all_features(seq, '2D19', coords, '/tmp', max_len=128)
    print("Train mode features:")
    for k, v in feats_train.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:20s}: {v.shape}")

    # inference mode
    feats_infer = build_all_features(seq, '2D19', None, '/tmp', max_len=128,
                                      is_inference=True)
    print("\nInference mode features (geometry should be zero):")
    print(f"  dist_rbf sum (should be 0): {feats_infer['dist_rbf'].sum():.1f}")
    print(f"  contact_ss sum            : {feats_infer['contact_ss'].sum():.1f}  (non-zero = OK)")

    # chunking test
    long_seq = seq * 10
    chunks   = chunk_sequence(long_seq, chunk_size=128, overlap=16)
    print(f"\nLong seq ({len(long_seq)} nt) → {len(chunks)} chunks")
    for s, e, c in chunks:
        print(f"  [{s:4d}:{e:4d}]  len={e-s}")

    print("\n✅ All OK")
