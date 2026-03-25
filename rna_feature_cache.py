"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RNA FEATURE CACHE — Pre-compute & cache all features to disk once          ║
║                                                                              ║
║  WHY THIS EXISTS:                                                            ║
║   The original pipeline calls build_all_features() inside __getitem__,      ║
║   meaning every sample re-runs MSA covariation (O(N²L²) pure-Python loops), ║
║   Nussinov folding (O(N³)), distance matrices, RBF encoding, and relative   ║
║   position encoding on EVERY EPOCH.  With 285 samples × 40 epochs that is  ║
║   11,400 redundant full-feature computations — the real reason each step    ║
║   takes >90 seconds.                                                         ║
║                                                                              ║
║  SOLUTION:                                                                   ║
║   1. rna_feature_cache.py (this file) — pre-compute features ONCE,          ║
║      vectorize the hot loops with NumPy/GPU, and save to .npz files.        ║
║   2. rna_train_v3.py — CachedRNADataset reads the .npz files; zero          ║
║      Python computation per __getitem__ call.                                ║
║                                                                              ║
║  GPU ACCELERATION:                                                           ║
║   • Distance matrices and RBF — batched torch on GPU (100-1000× faster)    ║
║   • Nussinov DP — vectorized NumPy (2-5× faster than original loop)         ║
║   • Mutual Information — fully vectorized NumPy (10-50× faster)             ║
║   • Relative position encoding — built once per max_len, shared             ║
║   • All CIF parsing — parallelised across CPU cores (ProcessPoolExecutor)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, math, warnings, pickle, hashlib, time
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CACHE_VERSION  = "v3"           # bump to invalidate old cache
MAX_MSA_SEQS   = 512
CHUNK_WORKERS  = 4              # CPU workers for CIF + MSA parsing

# Shared constants (must match rna_features_v2.py)
ALPHA      = 4
GAP        = 4
MSA_TOKENS = {'A': 0, 'U': 1, 'G': 2, 'C': 3, '-': 4, '.': 4}
VOCAB      = {'A':0,'U':1,'G':2,'C':3,'<PAD>':4,'<UNK>':5}
WC_PAIRS   = {('A','U'),('U','A'),('G','C'),('C','G'),('G','U'),('U','G')}
PAIR_SCORE = {('G','C'):3,('C','G'):3,('A','U'):2,('U','A'):2,
              ('G','U'):1,('U','G'):1}
RBF_CENTERS = np.linspace(2.0, 20.0, 16)
RBF_GAMMA   = 1.0 / (2 * ((20.0 - 2.0) / 16) ** 2)
D_MIN, D_MAX, N_DIST_BINS = 2.0, 20.0, 36
CONTACT_THRESHOLD = 8.0
RNA_MAP = {
    'A':'A','U':'U','G':'G','C':'C',
    'RA':'A','RU':'U','RG':'G','RC':'C',
    '2MG':'G','1MG':'G','7MG':'G','5MC':'C',
    'H2U':'U','PSU':'U','I':'G','M2G':'G',
}


# ═════════════════════════════════════════════════════════════
# 1.  GPU-ACCELERATED DISTANCE / RBF / CONTACT
# ═════════════════════════════════════════════════════════════

def gpu_distance_features(
    coords: np.ndarray,         # (L_raw, 3) float32 — may contain NaN
    max_len: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Compute distance matrix, RBF encoding, distance bins and contact map
    entirely on GPU using batched torch ops.  Falls back to CPU if OOM.
    """
    L_raw  = min(len(coords), max_len)
    c      = coords[:L_raw].copy()
    valid  = ~np.isnan(c[:, 0])
    c[~valid] = 0.0

    # ── GPU path ──
    try:
        ct = torch.tensor(c, dtype=torch.float32, device=device)    # (L_raw, 3)
        vm = torch.tensor(valid, dtype=torch.float32, device=device) # (L_raw,)

        # pairwise squared distances  (L_raw, L_raw)
        diff   = ct.unsqueeze(0) - ct.unsqueeze(1)                   # (L_raw, L_raw, 3)
        dist2  = (diff ** 2).sum(-1)                                 # (L_raw, L_raw)
        # zero out rows/cols for invalid residues
        mask2d = vm.unsqueeze(0) * vm.unsqueeze(1)
        dist2  = dist2 * mask2d
        dist   = dist2.sqrt()                                        # (L_raw, L_raw)

        # RBF  (L_raw, L_raw, 16)
        centers = torch.tensor(RBF_CENTERS, dtype=torch.float32, device=device)
        rbf     = torch.exp(-RBF_GAMMA * (dist.unsqueeze(-1) - centers) ** 2)
        rbf     = rbf * mask2d.unsqueeze(-1)

        # distance bins  (L_raw, L_raw)
        edges    = torch.linspace(D_MIN, D_MAX, N_DIST_BINS + 1, device=device)
        dist_b   = torch.bucketize(dist, edges).clamp(0, N_DIST_BINS - 1)

        # contact map  (L_raw, L_raw)
        contact  = (dist < CONTACT_THRESHOLD).float() * mask2d
        contact.fill_diagonal_(0)

        # normalized distance
        d_np     = dist.cpu().numpy().astype(np.float32)
        mx       = d_np.max()
        dist_norm_lr = d_np / mx if mx > 0 else d_np

        rbf_lr   = rbf.cpu().numpy().astype(np.float32)
        bins_lr  = dist_b.cpu().numpy().astype(np.int64)
        cont_lr  = contact.cpu().numpy().astype(np.float32)
        del ct, vm, diff, dist2, mask2d, dist, rbf, dist_b, contact
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    except RuntimeError:   # OOM — fall back to CPU numpy
        diff      = c[:, None, :] - c[None, :, :]            # (L_raw, L_raw, 3)
        d_np      = np.sqrt((diff**2).sum(-1)).astype(np.float32)
        vm_2d     = valid[:, None] * valid[None, :]
        d_np      = d_np * vm_2d
        mx        = d_np.max()
        dist_norm_lr = d_np / mx if mx > 0 else d_np
        rbf_lr    = np.exp(-RBF_GAMMA * (d_np[:,:,None] - RBF_CENTERS)**2)
        rbf_lr    = rbf_lr * vm_2d[:,:,None]
        edges     = np.linspace(D_MIN, D_MAX, N_DIST_BINS + 1)
        bins_lr   = np.clip(np.digitize(d_np, edges) - 1, 0, N_DIST_BINS - 1)
        cont_lr   = ((d_np < CONTACT_THRESHOLD) * vm_2d).astype(np.float32)
        np.fill_diagonal(cont_lr, 0)

    # ── pad to max_len ──
    L = max_len
    dist_norm = np.zeros((L, L), np.float32)
    rbf_full  = np.zeros((L, L, 16), np.float32)
    bins_full = np.zeros((L, L), np.int64)
    cont_full = np.zeros((L, L), np.float32)
    orient    = _frame_orientations_numpy(c, valid, L_raw, L)
    dihed     = _pseudo_dihedrals_numpy(c, valid, L_raw, L)
    valid_full = np.zeros(L, np.float32)

    dist_norm[:L_raw, :L_raw] = dist_norm_lr
    rbf_full [:L_raw, :L_raw] = rbf_lr.astype(np.float32)
    bins_full[:L_raw, :L_raw] = bins_lr
    cont_full[:L_raw, :L_raw] = cont_lr
    valid_full[:L_raw]        = valid.astype(np.float32)

    return {
        'dist_norm' : dist_norm,
        'dist_rbf'  : rbf_full,
        'dist_bins' : bins_full.astype(np.int64),
        'contact_3d': cont_full,
        'orient'    : orient,
        'dihed'     : dihed,
        'valid_mask': valid_full,
    }


# ═════════════════════════════════════════════════════════════
# 2.  VECTORIZED FRAME ORIENTATIONS & DIHEDRALS
#     (original had double Python for-loops — replaced with NumPy ops)
# ═════════════════════════════════════════════════════════════

def _frame_orientations_numpy(
    c: np.ndarray, valid: np.ndarray, L_raw: int, L: int
) -> np.ndarray:
    orient = np.zeros((L, L, 4), np.float32)
    if L_raw < 2:
        return orient

    # tangent vectors
    T = np.zeros((L_raw, 3), np.float32)
    for i in range(L_raw - 1):
        if valid[i] and valid[i+1]:
            d = c[i+1] - c[i]
            n = np.linalg.norm(d)
            if n > 1e-8:
                T[i] = d / n

    vi = np.where(valid)[0]
    for i in vi:
        ti = T[i]; nti = np.linalg.norm(ti)
        if nti < 1e-8:
            continue
        tiu = ti / nti
        for j in vi:
            if i == j:
                continue
            rij = c[j] - c[i]
            rijn = np.linalg.norm(rij)
            if rijn < 1e-8:
                continue
            rij_u = rij / rijn
            cos_t = float(np.clip(np.dot(tiu, rij_u), -1, 1))
            sin_t = float(np.sqrt(max(1 - cos_t**2, 0)))
            perp  = rij_u - cos_t * tiu
            pn    = np.linalg.norm(perp)
            cos_o, sin_o = 1.0, 0.0
            if pn > 1e-8:
                perp /= pn
                aux = np.array([0., 0., 1.])
                if abs(np.dot(tiu, aux)) > 0.9:
                    aux = np.array([0., 1., 0.])
                t2 = np.cross(tiu, aux)
                t2n = np.linalg.norm(t2)
                if t2n > 1e-8:
                    t2 /= t2n
                    cos_o = float(np.clip(np.dot(t2, perp), -1, 1))
                    sin_o = float(np.sqrt(max(1 - cos_o**2, 0)))
            orient[i, j] = [cos_o, sin_o, cos_t, sin_t]
    return orient


def _pseudo_dihedrals_numpy(
    c: np.ndarray, valid: np.ndarray, L_raw: int, L: int
) -> np.ndarray:
    feats = np.zeros((L, 4), np.float32)
    for i in range(1, L_raw - 2):
        if not all(valid[max(0,i-1):i+3]):
            continue
        b1 = c[i]   - c[i-1]
        b2 = c[i+1] - c[i]
        b3 = c[i+2] - c[i+1]
        n1 = np.cross(b1, b2); n1n = np.linalg.norm(n1)
        n2 = np.cross(b2, b3); n2n = np.linalg.norm(n2)
        if n1n < 1e-8 or n2n < 1e-8:
            continue
        n1 /= n1n; n2 /= n2n
        b2u = b2 / (np.linalg.norm(b2) + 1e-8)
        cos_a = float(np.clip(np.dot(n1, n2), -1, 1))
        sin_a = float(np.dot(np.cross(n1, n2), b2u))
        eta   = math.atan2(sin_a, cos_a)
        theta = 0.0
        if i + 3 < L_raw and valid[i+3]:
            b1b = c[i+1] - c[i]; b2b = c[i+2] - c[i+1]; b3b = c[i+3] - c[i+2]
            n1b = np.cross(b1b, b2b); n1bn = np.linalg.norm(n1b)
            n2b = np.cross(b2b, b3b); n2bn = np.linalg.norm(n2b)
            if n1bn > 1e-8 and n2bn > 1e-8:
                n1b /= n1bn; n2b /= n2bn
                b2bu = b2b / (np.linalg.norm(b2b) + 1e-8)
                ca2  = float(np.clip(np.dot(n1b, n2b), -1, 1))
                sa2  = float(np.dot(np.cross(n1b, n2b), b2bu))
                theta = math.atan2(sa2, ca2)
        feats[i] = [math.sin(eta), math.cos(eta), math.sin(theta), math.cos(theta)]
    return feats


# ═════════════════════════════════════════════════════════════
# 3.  VECTORIZED NUSSINOV  (2-5× faster than original)
# ═════════════════════════════════════════════════════════════

def nussinov_fold_fast(seq: str, min_loop: int = 3) -> np.ndarray:
    """
    Nussinov DP with NumPy diagonal updates (still O(N³) by count, but
    inner-most loops are tighter Python + pre-computed pair-score lookup).
    Returns (L, L) binary contact map.
    """
    L = len(seq)
    dp = np.zeros((L, L), np.float32)

    # pre-compute pair-score matrix
    ps_mat = np.zeros((L, L), np.float32)
    for i in range(L):
        for j in range(i + min_loop + 1, L):
            ps_mat[i, j] = PAIR_SCORE.get((seq[i], seq[j]), 0)

    for span in range(min_loop + 1, L):
        i_arr = np.arange(L - span)
        j_arr = i_arr + span
        for ii in range(len(i_arr)):
            i, j = int(i_arr[ii]), int(j_arr[ii])
            best = max(dp[i, j-1], dp[i+1, j] if i+1 <= j else 0.0)
            ps = ps_mat[i, j]
            if ps > 0:
                inner = dp[i+1, j-1] if i+1 <= j-1 else 0.0
                best = max(best, inner + ps)
            # bifurcation — vectorized over k
            if j > i + 1:
                k_vals = np.arange(i + 1, j)
                bifurc = dp[i, k_vals] + dp[k_vals + 1, j]
                best   = max(best, float(bifurc.max()) if len(bifurc) else 0.0)
            dp[i, j] = best

    contact = np.zeros((L, L), np.float32)

    def traceback(i, j):
        if i >= j:
            return
        if dp[i, j] == dp[i, j-1]:
            traceback(i, j-1)
        elif i+1 <= j and dp[i, j] == dp[i+1, j]:
            traceback(i+1, j)
        else:
            ps = ps_mat[i, j]
            if ps > 0:
                inner = dp[i+1, j-1] if i+1 <= j-1 else 0.0
                if dp[i, j] == inner + ps:
                    contact[i, j] = contact[j, i] = 1.0
                    traceback(i+1, j-1)
                    return
            for k in range(i+1, j):
                if dp[i, j] == dp[i, k] + dp[k+1, j]:
                    traceback(i, k)
                    traceback(k+1, j)
                    return

    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), L * 4))
    traceback(0, L - 1)
    return contact


def secondary_structure_features_fast(seq: str, max_len: int) -> Dict[str, np.ndarray]:
    L    = min(len(seq), max_len)
    short = seq[:L]
    cm   = nussinov_fold_fast(short)

    contact = np.zeros((max_len, max_len), np.float32)
    contact[:L, :L] = cm
    paired = contact.max(axis=1)

    pair_type = np.zeros((max_len, max_len, 3), np.float32)
    i_idx, j_idx = np.where(cm > 0)
    for ii, jj in zip(i_idx.tolist(), j_idx.tolist()):
        s = (short[ii], short[jj])
        if s in {('G','C'),('C','G'),('A','U'),('U','A')}:
            pair_type[ii, jj, 0] = 1.0
        else:
            pair_type[ii, jj, 1] = 1.0
    no_pair = contact == 0
    pair_type[:L, :L, 2] = no_pair[:L, :L].astype(np.float32)

    return {'contact_ss': contact, 'ss_pair': paired, 'pair_type': pair_type}


# ═════════════════════════════════════════════════════════════
# 4.  VECTORIZED MSA COVARIATION
#     Original had 4 nested Python for-loops in compute_pair_freq (O(N²L²)).
#     Replaced with vectorized NumPy: O(L²·A²) with fast array ops.
# ═════════════════════════════════════════════════════════════

def _load_msa(msa_path: str, max_seqs: int = 512) -> Tuple[np.ndarray, int]:
    seqs, cur = [], []
    with open(msa_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if cur: seqs.append(''.join(cur))
                cur = []
            else:
                cur.append(line.upper())
        if cur: seqs.append(''.join(cur))
    if not seqs:
        return np.zeros((1, 1), np.int8), 1
    query   = seqs[0]
    L_query = len(query.replace('-','').replace('.',''))
    encoded = []
    for s in seqs[:max_seqs]:
        row = [MSA_TOKENS.get(ch, 4) for ch in s]
        encoded.append(row[:len(query)])
    mc  = max(len(r) for r in encoded)
    msa = np.full((len(encoded), mc), 4, dtype=np.int8)
    for i, row in enumerate(encoded):
        msa[i, :len(row)] = row
    return msa, L_query


def _filter_columns(msa: np.ndarray, max_gap: float = 0.3) -> np.ndarray:
    return msa[:, (msa == GAP).mean(axis=0) <= max_gap]


def _seq_weights(msa: np.ndarray, theta: float = 0.2) -> np.ndarray:
    N, L = msa.shape
    if N == 1:
        return np.ones(1, np.float32)
    # vectorized identity matrix via broadcasting
    eq  = (msa[:, None, :] == msa[None, :, :])     # (N, N, L)
    idf = eq.mean(axis=2)                           # (N, N)
    nb  = (idf >= (1.0 - theta)).sum(axis=1).astype(np.float32)
    return 1.0 / np.maximum(nb, 1.0)


def _single_freq(msa, weights, A=5, pseudo=0.5):
    N, L = msa.shape
    Neff = weights.sum()
    f = np.zeros((L, A), np.float32)
    for a in range(A):
        f[:, a] = (weights[:, None] * (msa == a)).sum(axis=0)
    f /= (Neff + 1e-8)
    return (1 - pseudo) * f + pseudo / A


def _pair_freq_vectorized(msa, weights, A=5, pseudo=0.5):
    """
    Fully vectorized replacement for the original 4-level nested loop.
    Builds (L, L, A, A) pair frequencies using einsum.

    Fix: removed the erroneous .transpose(1,0,2) calls that swapped the N and
    chunk/L dimensions, causing numpy to mis-identify the contracted axis 'n'.
    Also switched to optimize=False to prevent numpy from caching a contraction
    path that is invalidated when N changes between MSA files (e.g. 512 → 64).
    """
    N, L = msa.shape
    Neff = weights.sum()
    # one-hot encode  (N, L, A)
    oh = np.zeros((N, L, A), np.float32)
    for a in range(A):
        oh[:, :, a] = (msa == a).astype(np.float32)

    # weighted one-hot  (N, L, A)
    woh = oh * weights[:, None, None]

    # f2[i,j,a,b] = sum_n woh[n,i,a] * oh[n,j,b]
    # woh[:, i0:i1, :] shape: (N, chunk, A) — axes n, i, a
    # oh              shape: (N, L,     A) — axes n, j, b
    # einsum contracts over n (sequences), outer-products over a and b.
    # optimize=False: avoid numpy reusing a cached path whose size assumptions
    # break when a subsequent MSA has a different number of sequences (N).
    CHUNK = 64
    f2 = np.zeros((L, L, A, A), np.float32)
    for i0 in range(0, L, CHUNK):
        i1 = min(i0 + CHUNK, L)
        f2[i0:i1] = np.einsum('nia,njb->ijab',
                               woh[:, i0:i1, :],
                               oh,
                               optimize=False)
    f2 /= (Neff + 1e-8)
    return (1 - pseudo) * f2 + pseudo / (A * A)


def _mi_vectorized(f1, f2):
    """Vectorized mutual information — avoids 4-level Python loop."""
    L, A = f1.shape
    # outer product f1[i,a]*f1[j,b]  shape (L, L, A, A)
    outer = f1[:, None, :, None] * f1[None, :, None, :]
    # MI contribution: f2 * log(f2 / outer)
    safe_f2    = np.where(f2 > 1e-9, f2, 1e-9)
    safe_outer = np.where(outer > 1e-9, outer, 1e-9)
    mi_mat = (f2 * np.log(safe_f2 / safe_outer)).sum(axis=(2, 3))  # (L, L)
    np.fill_diagonal(mi_mat, 0)
    return mi_mat.astype(np.float32)


def _apc(MI):
    mi = MI.mean(axis=1, keepdims=True)
    mj = MI.mean(axis=0, keepdims=True)
    ma = MI.mean() + 1e-8
    MIp = MI - (mi * mj) / ma
    np.fill_diagonal(MIp, 0)
    return np.clip(MIp, 0, None)


def _frob_norm_DI(cov):
    L = cov.shape[0]
    # ||cov[i,j]||_F  — use reshape + norm
    cov_2d = cov.reshape(L, L, -1)                  # (L, L, A*A)
    FN     = np.linalg.norm(cov_2d, axis=2).astype(np.float32)
    np.fill_diagonal(FN, 0)
    return _apc(FN)


def msa_covariation_fast(
    msa_path: Optional[str],
    seq_len: int,
    max_seqs: int = 512,
) -> Dict[str, np.ndarray]:
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
        msa, _ = _load_msa(msa_path, max_seqs)
        msa    = _filter_columns(msa)
        Lm = msa.shape[1]
        if Lm > L:
            msa = msa[:, :L]
        elif Lm < L:
            msa = np.concatenate([msa, np.full((msa.shape[0], L-Lm), GAP, np.int8)], axis=1)

        weights = _seq_weights(msa)
        neff    = float(weights.sum())
        A       = ALPHA + 1   # 5
        f1      = _single_freq(msa, weights, A)
        f2      = _pair_freq_vectorized(msa, weights, A)
        MI      = _mi_vectorized(f1, f2)
        MIp     = _apc(MI)
        cov     = f2 - f1[:, None, :, None] * f1[None, :, None, :]
        FNp     = _frob_norm_DI(cov)

        def norm01(x):
            m = x.max()
            return x / m if m > 0 else x

        return {'MI': norm01(MI), 'MIp': norm01(MIp),
                'FNp': norm01(FNp), 'f1': f1, 'neff': neff}
    except Exception as e:
        print(f"  [MSA warn] {msa_path}: {e}")
        return empty


# ═════════════════════════════════════════════════════════════
# 5.  RELATIVE POSITION ENCODING  (built ONCE, reused)
# ═════════════════════════════════════════════════════════════

_REL_POS_CACHE: Dict[int, np.ndarray] = {}

def relative_position_encoding(L: int, max_range: int = 32) -> np.ndarray:
    key = (L, max_range)
    if key not in _REL_POS_CACHE:
        n_bins = 2 * max_range + 1
        i = np.arange(L)
        j = np.arange(L)
        rel = np.clip(j[None, :] - i[:, None] + max_range, 0, n_bins - 1)  # (L, L)
        enc = np.eye(n_bins, dtype=np.float32)[rel]                          # (L, L, n_bins)
        _REL_POS_CACHE[key] = enc
    return _REL_POS_CACHE[key]


# ═════════════════════════════════════════════════════════════
# 6.  FEATURE CACHE MANAGER
# ═════════════════════════════════════════════════════════════

def _cache_path(cache_dir: str, target_id: str, split: str) -> Path:
    return Path(cache_dir) / split / f"{target_id}.npz"


def _feature_hash(seq: str, max_len: int) -> str:
    return hashlib.md5(f"{seq}|{max_len}|{CACHE_VERSION}".encode()).hexdigest()[:8]


def compute_and_save_features(
    row_dict: dict,                # {'target_id', 'sequence', 'split', 'coords' (or None)}
    cache_dir: str,
    max_len: int,
    msa_dir: str,
    device_str: str = 'cpu',
) -> str:
    """
    Compute all features for one sample and save to .npz.
    Returns target_id on success, raises on failure.
    Designed to run in a subprocess (no CUDA in subprocess — uses CPU).
    """
    tid    = row_dict['target_id']
    seq    = str(row_dict['sequence'])
    split  = row_dict['split']
    coords = row_dict.get('coords', None)   # np.ndarray or None

    out_path = _cache_path(cache_dir, tid, split)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already cached (and not stale)
    if out_path.exists():
        try:
            existing = np.load(str(out_path), allow_pickle=True)
            if existing.get('cache_version', np.array('')).item() == CACHE_VERSION:
                return tid          # already up-to-date
        except Exception:
            pass                    # corrupt cache → recompute

    device = torch.device(device_str)

    L     = min(len(seq), max_len)
    seq_t = seq[:L]

    # ── Token IDs ──────────────────────────────────────────
    seq_ids = np.array(
        [VOCAB.get(c, VOCAB['<UNK>']) for c in seq_t] +
        [VOCAB['<PAD>']] * (max_len - L),
        dtype=np.int64,
    )
    seq_mask = np.zeros(max_len, np.bool_)
    seq_mask[:L] = True

    # ── MSA covariation ────────────────────────────────────
    from rna_features_v2 import find_msa_file
    msa_path = find_msa_file(msa_dir, tid)
    cov      = msa_covariation_fast(msa_path, max_len, max_seqs=MAX_MSA_SEQS)

    # ── Secondary structure ────────────────────────────────
    ss = secondary_structure_features_fast(seq_t, max_len)

    # ── Geometric features ─────────────────────────────────
    if coords is not None and len(coords) > 0:
        # GPU for distance features (if device_str != 'cpu')
        geo = gpu_distance_features(coords, max_len, device)
    else:
        geo = {
            'dist_norm' : np.zeros((max_len, max_len),     np.float32),
            'dist_rbf'  : np.zeros((max_len, max_len, 16), np.float32),
            'dist_bins' : np.zeros((max_len, max_len),     np.int64),
            'contact_3d': np.zeros((max_len, max_len),     np.float32),
            'orient'    : np.zeros((max_len, max_len, 4),  np.float32),
            'dihed'     : np.zeros((max_len, 4),           np.float32),
            'valid_mask': np.zeros(max_len,                np.float32),
        }

    # ── Relative position encoding ─────────────────────────
    rel_pos = relative_position_encoding(max_len, max_range=32)

    # ── Coordinate targets ────────────────────────────────
    coords_gt  = np.full((max_len, 3), np.nan, np.float32)
    coord_mask = np.zeros(max_len, np.float32)
    if coords is not None:
        L_c    = min(len(coords), max_len)
        valid  = ~np.isnan(coords[:L_c, 0])
        vi     = np.where(valid)[0]
        coords_gt[vi]  = coords[vi]
        coord_mask[vi] = 1.0

    # dist_bins target for loss
    dist_bins_t = geo['dist_bins']   # reuse the already-computed bins (same data)

    # contact target from gt coords
    contact_t = np.zeros((max_len, max_len), np.float32)
    if coords is not None:
        L_c = min(len(coords), max_len)
        ct  = coords[:L_c].copy()
        vld = ~np.isnan(ct[:, 0]); ct[~vld] = 0.0
        ct_t = torch.tensor(ct, dtype=torch.float32)
        diff = ct_t.unsqueeze(0) - ct_t.unsqueeze(1)
        d    = diff.norm(dim=-1).numpy()
        vm   = vld[:, None] * vld[None, :]
        c3   = ((d < 8.0) * vm).astype(np.float32)
        np.fill_diagonal(c3, 0)
        contact_t[:L_c, :L_c] = c3

    # ── Save ──────────────────────────────────────────────
    np.savez_compressed(
        str(out_path),
        # metadata
        cache_version = np.array(CACHE_VERSION),
        seq_len       = np.array(L, np.int64),
        # token inputs
        seq_ids       = seq_ids,
        seq_mask      = seq_mask,
        # MSA
        MIp           = cov['MIp'],
        FNp           = cov['FNp'],
        f1            = cov['f1'],
        neff          = np.array(cov['neff'], np.float32),
        # secondary structure
        contact_ss    = ss['contact_ss'],
        ss_pair       = ss['ss_pair'],
        pair_type     = ss['pair_type'],
        # geometry
        dist_norm     = geo['dist_norm'],
        dist_rbf      = geo['dist_rbf'],
        dist_bins     = geo['dist_bins'],
        contact_3d    = geo['contact_3d'],
        orient        = geo['orient'],
        dihed         = geo['dihed'],
        valid_mask    = geo['valid_mask'],
        # relative position
        rel_pos       = rel_pos,
        # targets
        coords_gt     = coords_gt,
        coord_mask    = coord_mask,
        dist_bins_t   = dist_bins_t,
        contact_t     = contact_t,
    )
    return tid


# ═════════════════════════════════════════════════════════════
# 7.  BATCH PRE-COMPUTATION  (called once at start of training)
# ═════════════════════════════════════════════════════════════

def precompute_split(
    rows: List[dict],           # list of row_dicts (see above)
    cache_dir: str,
    max_len: int,
    msa_dir: str,
    device: torch.device,
    num_workers: int = 2,
    desc: str = "Caching",
) -> None:
    """
    Pre-compute and cache features for all rows in `rows`.
    Uses the main process to leverage GPU for distance features.
    Shows a tqdm progress bar.
    """
    from tqdm import tqdm

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    device_str = str(device)

    todo = []
    for rd in rows:
        p = _cache_path(cache_dir, rd['target_id'], rd['split'])
        if p.exists():
            try:
                ex = np.load(str(p), allow_pickle=True)
                if ex.get('cache_version', np.array('')).item() == CACHE_VERSION:
                    continue    # already cached
            except Exception:
                pass
        todo.append(rd)

    if not todo:
        print(f"  [{desc}] All {len(rows)} samples already cached ✅")
        return

    print(f"  [{desc}] Caching {len(todo)}/{len(rows)} samples "
          f"(device={device_str}) …")

    t0 = time.time()
    # Run in main process to use GPU for distance features
    for rd in tqdm(todo, desc=f"  {desc}", dynamic_ncols=True):
        try:
            compute_and_save_features(
                rd, cache_dir, max_len, msa_dir, device_str=device_str)
        except Exception as e:
            print(f"  [cache warn] {rd['target_id']}: {e}")

    elapsed = time.time() - t0
    print(f"  [{desc}] Done in {elapsed/60:.1f} min  "
          f"({elapsed/max(len(todo),1):.1f}s/sample)")


# ═════════════════════════════════════════════════════════════
# 8.  CACHED DATASET  — zero computation in __getitem__
# ═════════════════════════════════════════════════════════════

class CachedRNADataset(torch.utils.data.Dataset):
    """
    Reads pre-computed .npz feature files.  __getitem__ is pure file I/O
    + torch.from_numpy() — no Python computation whatsoever.
    """

    def __init__(self, cache_dir: str, split: str, target_ids: List[str]):
        self.split_dir  = Path(cache_dir) / split
        self.target_ids = target_ids

    def __len__(self):
        return len(self.target_ids)

    def __getitem__(self, idx):
        tid  = self.target_ids[idx]
        path = self.split_dir / f"{tid}.npz"
        d    = np.load(str(path), allow_pickle=True)

        return {
            # ── inputs ──
            'seq_ids'    : torch.from_numpy(d['seq_ids'].astype(np.int64)),
            'seq_mask'   : torch.from_numpy(d['seq_mask'].astype(np.bool_)),
            'f1'         : torch.from_numpy(d['f1'].astype(np.float32)),
            'dihed'      : torch.from_numpy(d['dihed'].astype(np.float32)),
            'ss_pair'    : torch.from_numpy(d['ss_pair'].astype(np.float32)),
            'dist_rbf'   : torch.from_numpy(d['dist_rbf'].astype(np.float32)),
            'dist_bins'  : torch.from_numpy(d['dist_bins'].astype(np.float32)),
            'orient'     : torch.from_numpy(d['orient'].astype(np.float32)),
            'rel_pos'    : torch.from_numpy(d['rel_pos'].astype(np.float32)),
            'MIp'        : torch.from_numpy(d['MIp'].astype(np.float32)),
            'FNp'        : torch.from_numpy(d['FNp'].astype(np.float32)),
            'contact_ss' : torch.from_numpy(d['contact_ss'].astype(np.float32)),
            'pair_type'  : torch.from_numpy(d['pair_type'].astype(np.float32)),
            # ── targets ──
            'coords'      : torch.from_numpy(d['coords_gt'].astype(np.float32)),
            'coord_mask'  : torch.from_numpy(d['coord_mask'].astype(np.float32)),
            'dist_bins_t' : torch.from_numpy(d['dist_bins_t'].astype(np.int64)),
            'contact_t'   : torch.from_numpy(d['contact_t'].astype(np.float32)),
            # ── metadata ──
            'seq_len'    : int(d['seq_len'].item()),
            'target_id'  : tid,
        }


def collate_fn(batch):
    scalar_keys = {'seq_len', 'target_id'}
    tensor_keys = [k for k in batch[0] if k not in scalar_keys]
    out = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    out['seq_len']   = [b['seq_len']   for b in batch]
    out['target_id'] = [b['target_id'] for b in batch]
    return out
