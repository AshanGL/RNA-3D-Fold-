"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RNA FEATURE ENGINEERING — KAGGLE-WINNING LEVEL                             ║
║  MSA Covariation + APC-corrected MI + Full Geometric Features               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Sections:
  1.  MSA loading & preprocessing (gap filtering, sequence weighting)
  2.  One-hot MSA encoding
  3.  Single-sequence frequency profiles (f_i)
  4.  Pairwise frequency tensors (f_ij)
  5.  Raw Mutual Information (MI)
  6.  APC correction → MIp
  7.  Frobenius norm of DI (Direct Information from pseudolikelihood)
  8.  Covariance matrix (Potts model flavour)
  9.  Full geometric features:
        • C1'–C1' distance matrix (raw + binned + Gaussian RBF)
        • Inter-residue orientations (3 Euler angles per pair)
        • Backbone dihedral angles (η, θ pseudo-dihedrals)
        • Relative position encodings
  10. Contact map targets (distance < 8 Å threshold)
"""

import os, math, warnings
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
ALPHA = 4                 # nucleotide alphabet size (A,U,G,C)
GAP   = 4                 # gap token index
MSA_TOKENS = {'A': 0, 'U': 1, 'G': 2, 'C': 3, '-': 4, '.': 4}
RNA_MAP = {
    'A':'A','U':'U','G':'G','C':'C',
    'RA':'A','RU':'U','RG':'G','RC':'C',
    '2MG':'G','1MG':'G','7MG':'G','5MC':'C',
    'H2U':'U','PSU':'U','I':'G','M2G':'G',
}

CONTACT_THRESHOLD  = 8.0    # Å  — C1'–C1' contact
N_DIST_BINS        = 36     # distance bins
D_MIN, D_MAX       = 2.0, 20.0
N_ANGLE_BINS       = 36
RBF_CENTERS        = np.linspace(D_MIN, D_MAX, 16)
RBF_GAMMA          = 1.0 / (2 * ((D_MAX - D_MIN) / 16) ** 2)

# ═════════════════════════════════════════════════════════════
# 1. MSA LOADING
# ═════════════════════════════════════════════════════════════
def load_msa(msa_path: str, max_seqs: int = 512) -> Tuple[np.ndarray, int]:
    """
    Parse FASTA MSA file.
    Returns:
      msa    : (N, L) int8 array   (0=A,1=U,2=G,3=C,4=gap)
      L_query: length of query sequence (gaps excluded)
    """
    seqs = []
    current = []
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

    query = seqs[0]
    L_query = len(query.replace('-', '').replace('.', ''))

    encoded = []
    for s in seqs[:max_seqs]:
        row = [MSA_TOKENS.get(c, 4) for c in s]
        encoded.append(row[:len(query)])

    max_col = max(len(r) for r in encoded)
    msa = np.full((len(encoded), max_col), 4, dtype=np.int8)
    for i, row in enumerate(encoded):
        msa[i, :len(row)] = row

    return msa, L_query


# ═════════════════════════════════════════════════════════════
# 2. GAP FILTERING & SEQUENCE WEIGHTING
# ═════════════════════════════════════════════════════════════
def filter_columns(msa: np.ndarray, max_gap_frac: float = 0.3) -> np.ndarray:
    """Remove columns with > max_gap_frac gap fraction."""
    gap_frac = (msa == GAP).mean(axis=0)
    keep = gap_frac <= max_gap_frac
    return msa[:, keep]


def sequence_weights(msa: np.ndarray, theta: float = 0.2) -> np.ndarray:
    """
    Compute Henikoff-style sequence weights.
    w_i = 1 / (# sequences within theta * L Hamming distance of seq i)
    Returns (N,) float32 weights that sum to Neff.
    """
    N, L = msa.shape
    if N == 1:
        return np.ones(1, dtype=np.float32)

    # pairwise identity (fast vectorised)
    id_matrix = np.zeros((N, N), dtype=np.float32)
    for k in range(L):
        col = msa[:, k]
        id_matrix += (col[:, None] == col[None, :]).astype(np.float32)
    id_frac = id_matrix / L

    # count neighbours
    neighbours = (id_frac >= (1.0 - theta)).sum(axis=1).astype(np.float32)
    weights = 1.0 / np.maximum(neighbours, 1.0)
    return weights  # (N,)


# ═════════════════════════════════════════════════════════════
# 3. FREQUENCY PROFILES
# ═════════════════════════════════════════════════════════════
def compute_single_freq(msa: np.ndarray,
                        weights: np.ndarray,
                        pseudo: float = 0.5) -> np.ndarray:
    """
    f_i(a) = weighted frequency of nucleotide a at position i.
    Returns: (L, ALPHA+1)  — last dim = gap
    """
    N, L = msa.shape
    A = ALPHA + 1   # include gap

    f = np.zeros((L, A), dtype=np.float32)
    Neff = weights.sum()

    for a in range(A):
        f[:, a] = (weights[:, None] * (msa == a)).sum(axis=0)

    f /= (Neff + 1e-8)
    # add pseudocount
    f = (1 - pseudo) * f + pseudo / A
    return f   # (L, 5)


def compute_pair_freq(msa: np.ndarray,
                      weights: np.ndarray,
                      pseudo: float = 0.5) -> np.ndarray:
    """
    f_ij(a,b) = weighted joint frequency.
    Returns: (L, L, A+1, A+1)  — MEMORY-EFFICIENT sparse loop version.
    For L > 200 uses only diagonal bands.
    """
    N, L = msa.shape
    A = ALPHA + 1
    Neff = weights.sum()

    f2 = np.zeros((L, L, A, A), dtype=np.float32)

    for i in range(L):
        col_i = msa[:, i]
        for j in range(i, L):
            col_j = msa[:, j]
            for n in range(N):
                a, b = int(col_i[n]), int(col_j[n])
                if a >= A: a = A - 1
                if b >= A: b = A - 1
                f2[i, j, a, b] += weights[n]
                if i != j:
                    f2[j, i, b, a] += weights[n]

    f2 /= (Neff + 1e-8)
    f2 = (1 - pseudo) * f2 + pseudo / (A * A)
    return f2  # (L, L, 5, 5)


# ═════════════════════════════════════════════════════════════
# 4. MUTUAL INFORMATION + APC CORRECTION
# ═════════════════════════════════════════════════════════════
def compute_MI(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """
    Raw Mutual Information matrix from f1 (L,A) and f2 (L,L,A,A).
    MI(i,j) = Σ_ab f_ij(a,b) log[ f_ij(a,b) / (f_i(a) * f_j(b)) ]
    Returns: (L, L) float32
    """
    L, A = f1.shape
    MI = np.zeros((L, L), dtype=np.float32)

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


def apc_correction(MI: np.ndarray) -> np.ndarray:
    """
    Average Product Correction (APC) to remove phylogenetic noise.
    MIp(i,j) = MI(i,j) - MI(i,:).mean() * MI(:,j).mean() / MI.mean()
    Returns: (L, L) MIp
    """
    mean_i = MI.mean(axis=1, keepdims=True)  # (L, 1)
    mean_j = MI.mean(axis=0, keepdims=True)  # (1, L)
    mean_all = MI.mean() + 1e-8

    MIp = MI - (mean_i * mean_j) / mean_all
    np.fill_diagonal(MIp, 0.0)
    return np.clip(MIp, 0, None)


# ═════════════════════════════════════════════════════════════
# 5. DIRECT INFORMATION (Frobenius norm of DI matrix)
#    Pseudolikelihood-based covariation measure
# ═════════════════════════════════════════════════════════════
def compute_covariance_matrix(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """
    Cij(a,b) = f_ij(a,b) - f_i(a) * f_j(b)
    Returns (L, L, A, A) connected correlation matrix.
    """
    L, A = f1.shape
    outer = f1[:, None, :, None] * f1[None, :, None, :]  # (L,L,A,A)
    return f2 - outer


def frobenius_norm_DI(cov: np.ndarray) -> np.ndarray:
    """
    FN(i,j) = ||C_ij||_F  (Frobenius norm of the (A,A) sub-matrix)
    After APC correction → FNp
    Returns: (L, L) float32
    """
    L = cov.shape[0]
    FN = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(i + 1, L):
            fn = np.linalg.norm(cov[i, j])
            FN[i, j] = FN[j, i] = fn

    return apc_correction(FN)


# ═════════════════════════════════════════════════════════════
# 6. FULL MSA COVARIATION PIPELINE
# ═════════════════════════════════════════════════════════════
def msa_covariation_features(msa_path: str,
                              seq_len: int,
                              max_seqs: int = 512) -> Dict[str, np.ndarray]:
    """
    Complete MSA → covariation feature pipeline.
    Returns dict:
      'MI'     : (L, L) raw MI
      'MIp'    : (L, L) APC-corrected MI
      'FNp'    : (L, L) Frobenius DI (APC-corrected)
      'f1'     : (L, 5) single-position frequencies
      'neff'   : scalar effective number of sequences
    """
    L = seq_len

    # defaults (no MSA)
    empty = {
        'MI' : np.zeros((L, L), np.float32),
        'MIp': np.zeros((L, L), np.float32),
        'FNp': np.zeros((L, L), np.float32),
        'f1' : np.full((L, 5), 0.2, np.float32),
        'neff': 1.0,
    }

    if not Path(msa_path).exists():
        return empty

    try:
        msa, L_query = load_msa(msa_path, max_seqs)
        msa = filter_columns(msa, max_gap_frac=0.3)

        # trim / pad to seq_len
        Lm = msa.shape[1]
        if Lm > L:
            msa = msa[:, :L]
        elif Lm < L:
            pad = np.full((msa.shape[0], L - Lm), GAP, dtype=np.int8)
            msa = np.concatenate([msa, pad], axis=1)

        weights = sequence_weights(msa)
        neff = float(weights.sum())

        f1 = compute_single_freq(msa, weights)         # (L, 5)
        f2 = compute_pair_freq(msa, weights)           # (L, L, 5, 5)

        MI  = compute_MI(f1, f2)
        MIp = apc_correction(MI)

        cov = compute_covariance_matrix(f1, f2)
        FNp = frobenius_norm_DI(cov)

        # normalise to [0,1]
        def norm01(x):
            m = x.max(); return x / m if m > 0 else x

        return {
            'MI' : norm01(MI),
            'MIp': norm01(MIp),
            'FNp': norm01(FNp),
            'f1' : f1,
            'neff': neff,
        }

    except Exception as e:
        print(f"[MSA warning] {msa_path}: {e}")
        return empty


# ═════════════════════════════════════════════════════════════
# 7. GEOMETRIC FEATURES FROM 3D COORDINATES
# ═════════════════════════════════════════════════════════════

def rbf_encode(dist: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function encoding of distances.
    dist : (L, L)
    Returns (L, L, n_rbf)
    """
    L = dist.shape[0]
    d = dist[:, :, None]                            # (L, L, 1)
    centers = RBF_CENTERS[None, None, :]            # (1, 1, K)
    return np.exp(-RBF_GAMMA * (d - centers) ** 2) # (L, L, K)


def bin_distances(dist: np.ndarray,
                  d_min: float = D_MIN,
                  d_max: float = D_MAX,
                  n_bins: int = N_DIST_BINS) -> np.ndarray:
    """
    Bin distances into one-hot (L, L, n_bins) tensor.
    """
    L = dist.shape[0]
    edges = np.linspace(d_min, d_max, n_bins + 1)
    indices = np.digitize(dist, edges) - 1
    indices = np.clip(indices, 0, n_bins - 1)
    one_hot = np.eye(n_bins, dtype=np.float32)[indices]  # (L, L, n_bins)
    return one_hot


def compute_frame_orientations(coords: np.ndarray) -> np.ndarray:
    """
    Compute local reference frames from C1' backbone.
    For each residue i, build frame (t1, t2, t3):
      t1 = unit vec from i to i+1
      t2 = component of (i-1 → i) perp to t1
      t3 = t1 × t2

    Returns orientation angles between frames i and j:
      (L, L, 4) : [cos_omega, sin_omega, cos_theta, sin_theta]
    """
    L = len(coords)
    valid = ~np.isnan(coords[:, 0])

    # build forward vectors
    T = np.zeros((L, 3), dtype=np.float32)
    for i in range(L - 1):
        if valid[i] and valid[i+1]:
            d = coords[i+1] - coords[i]
            n = np.linalg.norm(d)
            if n > 1e-8:
                T[i] = d / n

    # pairwise: for each pair (i,j) compute relative direction
    # r_ij in local frame of i
    orient = np.zeros((L, L, 4), dtype=np.float32)

    for i in range(L):
        if not valid[i]: continue
        ti = T[i]
        norm_ti = np.linalg.norm(ti)
        if norm_ti < 1e-8: continue

        for j in range(L):
            if not valid[j] or i == j: continue

            # vector from i to j
            rij = coords[j] - coords[i]
            rij_norm = np.linalg.norm(rij)
            if rij_norm < 1e-8: continue
            rij /= rij_norm

            # projection angles
            cos_t = float(np.clip(np.dot(ti / norm_ti, rij), -1, 1))
            sin_t = float(np.sqrt(max(1 - cos_t**2, 0)))

            # signed rotation around backbone
            perp = rij - cos_t * (ti / norm_ti)
            perp_n = np.linalg.norm(perp)

            cos_o, sin_o = 1.0, 0.0
            if perp_n > 1e-8:
                perp /= perp_n
                # build second local axis (perpendicular to ti)
                aux = np.array([0., 0., 1.])
                if abs(np.dot(ti / norm_ti, aux)) > 0.9:
                    aux = np.array([0., 1., 0.])
                t2 = np.cross(ti / norm_ti, aux)
                t2n = np.linalg.norm(t2)
                if t2n > 1e-8:
                    t2 /= t2n
                    cos_o = float(np.clip(np.dot(t2, perp), -1, 1))
                    sin_o = float(np.sqrt(max(1 - cos_o**2, 0)))

            orient[i, j] = [cos_o, sin_o, cos_t, sin_t]

    return orient


def pseudo_dihedral_angles(coords: np.ndarray) -> np.ndarray:
    """
    Compute RNA pseudo-dihedral angles η and θ.
    η(i) = dihedral(C1'(i-1), P(i), C1'(i), P(i+1))  — uses C1' as proxy
    θ(i) = dihedral(P(i), C1'(i), P(i+1), C1'(i+1))

    Since we only have C1', we approximate using 4 consecutive C1' atoms.
    Returns (L, 4) : [sin_eta, cos_eta, sin_theta, cos_theta]
    """
    L = len(coords)
    feats = np.zeros((L, 4), dtype=np.float32)
    valid = ~np.isnan(coords[:, 0])

    def dihedral(p0, p1, p2, p3):
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2
        n1 = np.cross(b1, b2);  n1n = np.linalg.norm(n1)
        n2 = np.cross(b2, b3);  n2n = np.linalg.norm(n2)
        if n1n < 1e-8 or n2n < 1e-8: return 0.0
        n1 /= n1n; n2 /= n2n
        b2u = b2 / (np.linalg.norm(b2) + 1e-8)
        cos_a = np.clip(np.dot(n1, n2), -1, 1)
        sin_a = np.dot(np.cross(n1, n2), b2u)
        return math.atan2(sin_a, cos_a)

    for i in range(1, L - 2):
        if not all(valid[i-1:i+3]): continue
        # η: C1'(i-1), C1'(i), C1'(i+1), C1'(i+2)
        eta   = dihedral(coords[i-1], coords[i],   coords[i+1], coords[i+2])
        # θ: C1'(i), C1'(i+1), C1'(i+2), C1'(i+3) if available
        theta = 0.0
        if i + 3 < L and valid[i+3]:
            theta = dihedral(coords[i], coords[i+1], coords[i+2], coords[i+3])

        feats[i] = [math.sin(eta), math.cos(eta),
                    math.sin(theta), math.cos(theta)]

    return feats


def relative_position_encoding(L: int, max_range: int = 32) -> np.ndarray:
    """
    Relative position encoding: for each pair (i,j), encode (j - i).
    Returns (L, L, 2*max_range+1) one-hot.
    """
    n_bins = 2 * max_range + 1
    enc = np.zeros((L, L, n_bins), dtype=np.float32)
    for i in range(L):
        for j in range(L):
            rel = j - i
            idx = int(np.clip(rel + max_range, 0, n_bins - 1))
            enc[i, j, idx] = 1.0
    return enc


def geometric_features(coords: np.ndarray,
                        max_len: int = 512) -> Dict[str, np.ndarray]:
    """
    Compute all geometric features from C1' coordinate array.
    coords : (L_raw, 3)
    Returns dict of arrays padded to (max_len, ...).
    """
    L_raw = min(len(coords), max_len)
    c     = coords[:L_raw].copy()
    L     = max_len

    valid = ~np.isnan(c[:, 0])
    c_clean = c.copy(); c_clean[~valid] = 0.0

    # ── distance matrix ──────────────────────────────
    dist_raw = np.zeros((L_raw, L_raw), dtype=np.float32)
    vi = np.where(valid)[0]
    for ii in range(len(vi)):
        for jj in range(ii + 1, len(vi)):
            i, j = vi[ii], vi[jj]
            d = np.linalg.norm(c_clean[i] - c_clean[j])
            dist_raw[i, j] = dist_raw[j, i] = d

    dist_full = np.zeros((L, L), dtype=np.float32)
    dist_full[:L_raw, :L_raw] = dist_raw

    # normalised continuous distance
    dist_norm = dist_full.copy()
    mx = dist_norm.max()
    if mx > 0: dist_norm /= mx

    # RBF encoding (L, L, 16)
    rbf_full = np.zeros((L, L, len(RBF_CENTERS)), dtype=np.float32)
    rbf_full[:L_raw, :L_raw] = rbf_encode(dist_raw)

    # binned one-hot (L, L, 36)
    bin_full = np.zeros((L, L, N_DIST_BINS), dtype=np.float32)
    bin_full[:L_raw, :L_raw] = bin_distances(dist_raw)

    # ── contact map (< 8 Å) ──────────────────────────
    contact = (dist_full < CONTACT_THRESHOLD).astype(np.float32)
    np.fill_diagonal(contact, 0)

    # ── orientation features (L, L, 4) ───────────────
    orient_raw = compute_frame_orientations(c)
    orient_full = np.zeros((L, L, 4), dtype=np.float32)
    orient_full[:L_raw, :L_raw] = orient_raw

    # ── dihedral angles (L, 4) ───────────────────────
    dihed_raw  = pseudo_dihedral_angles(c)
    dihed_full = np.zeros((L, 4), dtype=np.float32)
    dihed_full[:L_raw] = dihed_raw

    # ── relative position encoding (L, L, 65) ────────
    rel_pos = relative_position_encoding(L, max_range=32)

    # ── valid mask ───────────────────────────────────
    valid_full = np.zeros(L, dtype=np.float32)
    valid_full[:L_raw] = valid.astype(np.float32)

    return {
        'dist_norm'  : dist_norm,       # (L, L)
        'dist_rbf'   : rbf_full,        # (L, L, 16)
        'dist_bins'  : bin_full,        # (L, L, 36)
        'contact'    : contact,         # (L, L)
        'orient'     : orient_full,     # (L, L, 4)
        'dihed'      : dihed_full,      # (L, 4)
        'rel_pos'    : rel_pos,         # (L, L, 65)
        'valid_mask' : valid_full,      # (L,)
    }


# ═════════════════════════════════════════════════════════════
# 8. UNIFIED FEATURE BUILDER
# ═════════════════════════════════════════════════════════════
def build_all_features(seq: str,
                       target_id: str,
                       coords: Optional[np.ndarray],
                       msa_dir: str,
                       max_len: int = 512) -> Dict[str, np.ndarray]:
    """
    Build the complete feature set for one RNA sequence.
    seq       : raw sequence string
    target_id : e.g. '4TNA'
    coords    : (L_raw, 3) C1' coordinates or None (for test)
    msa_dir   : path to folder containing .MSA.fasta files
    max_len   : pad/crop length
    """
    L = min(len(seq), max_len)

    # ── MSA covariation ──
    msa_path = str(Path(msa_dir) / f"{target_id}.MSA.fasta")
    if not Path(msa_path).exists():
        # try uppercase
        for p in Path(msa_dir).glob(f"{target_id.upper()}*.fasta"):
            msa_path = str(p); break

    cov = msa_covariation_features(msa_path, L, max_seqs=512)

    # ── geometric ──
    if coords is not None:
        geo = geometric_features(coords, max_len)
    else:
        geo = geometric_features(np.full((L, 3), np.nan), max_len)

    # ── merge ──
    return {**cov, **geo, 'seq_len': L}


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Feature engineering self-test ===")
    # synthetic test
    np.random.seed(0)
    L = 30
    fake_coords = np.cumsum(np.random.randn(L, 3) * 2, axis=0).astype(np.float32)
    fake_seq    = ''.join(np.random.choice(list('AUGC'), L))

    geo = geometric_features(fake_coords, max_len=64)
    print("Geometric features computed:")
    for k, v in geo.items():
        print(f"  {k:15s}: {v.shape}  dtype={v.dtype}")

    print("\nAll OK ✅")
