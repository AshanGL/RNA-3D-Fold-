"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           PHYSICS REFINEMENT & POST-PROCESSING MODULE                       ║
║  Applies structural constraints to predicted 3D RNA coordinates              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Modules:
  1. Bond-length enforcement (P–C1' backbone)
  2. Clash removal (steric repulsion)
  3. Gradient-based coordinate refinement (energy minimization)
  4. Multi-model ensemble averaging
  5. Evaluation metrics (TM-score, RMSD, GDT)
  6. Final submission formatter
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Tuple

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
# Ideal RNA backbone distances (Angstroms)
IDEAL_P_C1  = 8.9    # P to C1' (across one residue)
IDEAL_C1_C1 = 6.0    # C1' to next C1' (stacking distance)
VDW_RADIUS  = 1.7    # van der Waals radius for clash
MIN_C1_DIST = 3.4    # minimum allowed C1'–C1' distance

# ─────────────────────────────────────────────────────────────
# 1. COORDINATE NORMALIZATION
# ─────────────────────────────────────────────────────────────
def center_and_scale(coords: np.ndarray,
                     target_mean_dist: float = 6.0) -> np.ndarray:
    """
    Center coordinates and scale so mean consecutive C1'-C1' distance
    matches ideal RNA stacking (~6 Å).
    """
    valid = ~np.isnan(coords[:, 0])
    if valid.sum() < 2:
        return coords

    c = coords.copy()
    c[valid] -= c[valid].mean(axis=0)  # center

    # compute mean consecutive distance
    dists = []
    idx = np.where(valid)[0]
    for i in range(len(idx) - 1):
        if idx[i+1] == idx[i] + 1:
            dists.append(np.linalg.norm(c[idx[i+1]] - c[idx[i]]))

    if dists:
        mean_d = np.mean(dists)
        if mean_d > 0:
            c[valid] *= (target_mean_dist / mean_d)

    return c


# ─────────────────────────────────────────────────────────────
# 2. BOND-LENGTH ENFORCEMENT
# ─────────────────────────────────────────────────────────────
def enforce_bond_lengths(coords: np.ndarray,
                         ideal_dist: float = IDEAL_C1_C1,
                         n_iter: int = 20,
                         alpha: float = 0.3) -> np.ndarray:
    """
    Iteratively nudge consecutive C1' atoms towards ideal distance.
    Uses a spring-like constraint.
    """
    c = coords.copy()
    L = len(c)
    valid = ~np.isnan(c[:, 0])

    for _ in range(n_iter):
        for i in range(L - 1):
            j = i + 1
            if not (valid[i] and valid[j]):
                continue

            diff = c[j] - c[i]
            dist = np.linalg.norm(diff) + 1e-8
            correction = (dist - ideal_dist) / dist * alpha
            delta = diff * correction * 0.5

            c[i] += delta
            c[j] -= delta

    return c


# ─────────────────────────────────────────────────────────────
# 3. CLASH REMOVAL
# ─────────────────────────────────────────────────────────────
def remove_clashes(coords: np.ndarray,
                   min_dist: float = MIN_C1_DIST,
                   n_iter: int = 30,
                   alpha: float = 0.5) -> np.ndarray:
    """
    Repel any pair of atoms that are closer than min_dist.
    Only applied to non-consecutive pairs (|i-j| > 2).
    """
    c = coords.copy()
    L = len(c)
    valid = ~np.isnan(c[:, 0])

    for _ in range(n_iter):
        clashes = 0
        for i in range(L):
            if not valid[i]:
                continue
            for j in range(i + 3, L):  # skip local pairs
                if not valid[j]:
                    continue
                diff = c[j] - c[i]
                dist = np.linalg.norm(diff) + 1e-8
                if dist < min_dist:
                    clashes += 1
                    repulsion = (min_dist - dist) / dist * alpha
                    delta = diff * repulsion * 0.5
                    c[i] -= delta
                    c[j] += delta
        if clashes == 0:
            break

    return c


# ─────────────────────────────────────────────────────────────
# 4. GRADIENT-BASED REFINEMENT (differentiable energy)
# ─────────────────────────────────────────────────────────────
class EnergyRefinement(nn.Module):
    """
    Differentiable energy function for coordinate refinement.
    Minimizes:
      E = w_bond * E_bond + w_clash * E_clash + w_pair * E_pair
    """
    def __init__(self,
                 w_bond: float = 1.0,
                 w_clash: float = 2.0,
                 w_pair: float = 0.5):
        super().__init__()
        self.w_bond  = w_bond
        self.w_clash = w_clash
        self.w_pair  = w_pair

    def bond_energy(self, coords):
        """Harmonic bond energy between consecutive residues."""
        diff = coords[1:] - coords[:-1]                      # (L-1, 3)
        dist = diff.norm(dim=-1)                              # (L-1,)
        return ((dist - IDEAL_C1_C1) ** 2).mean()

    def clash_energy(self, coords):
        """Soft-sphere clash energy for all pairs |i-j|>2."""
        L = coords.shape[0]
        E = coords.new_zeros(1)
        for i in range(L):
            j_start = i + 3
            if j_start >= L:
                break
            diff = coords[j_start:] - coords[i].unsqueeze(0)   # (L-i-3, 3)
            dist = diff.norm(dim=-1) + 1e-8
            clash = F.relu(MIN_C1_DIST - dist)
            E = E + (clash ** 2).sum()
        return E / max(L, 1)

    def pair_energy(self, coords, contact_map):
        """Attract paired residues (contact_map = 1) towards IDEAL_C1_C1 * 1.5."""
        if contact_map is None:
            return coords.new_zeros(1)
        target_dist = IDEAL_C1_C1 * 1.5
        E = coords.new_zeros(1)
        pairs = (contact_map > 0.5).nonzero(as_tuple=False)
        for p in pairs:
            i, j = p[0].item(), p[1].item()
            if abs(i - j) < 3:
                continue
            diff = coords[i] - coords[j]
            dist = diff.norm() + 1e-8
            E = E + (dist - target_dist) ** 2
        return E / max(len(pairs), 1)

    def forward(self, coords, contact_map=None):
        import torch.nn.functional as F
        return (self.w_bond  * self.bond_energy(coords) +
                self.w_clash * self.clash_energy(coords) +
                self.w_pair  * self.pair_energy(coords, contact_map))


import torch.nn.functional as F   # needed for relu above

def gradient_refine(coords: np.ndarray,
                    contact_map: Optional[np.ndarray] = None,
                    n_steps: int = 100,
                    lr: float = 0.01) -> np.ndarray:
    """
    Optimize coordinates using gradient descent on physics energy.
    coords     : (L, 3) numpy
    contact_map: (L, L) numpy or None
    Returns refined (L, 3) numpy.
    """
    valid = ~np.isnan(coords[:, 0])
    if valid.sum() < 3:
        return coords

    c = coords.copy()
    c[~valid] = 0.0

    tc = torch.tensor(c[valid], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([tc], lr=lr)
    energy_fn = EnergyRefinement()

    cm = None
    if contact_map is not None:
        cm = torch.tensor(contact_map[valid][:, valid], dtype=torch.float32)

    for _ in range(n_steps):
        opt.zero_grad()
        loss = energy_fn(tc, cm)
        loss.backward()
        opt.step()

    result = c.copy()
    result[valid] = tc.detach().numpy()
    return result


# ─────────────────────────────────────────────────────────────
# 5. FULL REFINEMENT PIPELINE
# ─────────────────────────────────────────────────────────────
def refine_structure(coords: np.ndarray,
                     contact_map: Optional[np.ndarray] = None,
                     use_gradient: bool = True) -> np.ndarray:
    """
    Full post-processing pipeline for a single structure.
    coords     : (L, 3) raw predicted
    contact_map: (L, L) optional
    Returns    : (L, 3) refined
    """
    c = center_and_scale(coords)
    c = enforce_bond_lengths(c)
    c = remove_clashes(c)
    if use_gradient:
        c = gradient_refine(c, contact_map, n_steps=50)
    c = enforce_bond_lengths(c, n_iter=10)  # final pass
    return c


# ─────────────────────────────────────────────────────────────
# 6. EVALUATION METRICS
# ─────────────────────────────────────────────────────────────
def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Kabsch RMSD between two (N, 3) arrays after optimal superposition.
    Returns (rmsd, rotated_P).
    """
    assert P.shape == Q.shape
    P = P - P.mean(0)
    Q = Q - Q.mean(0)

    H  = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d  = np.linalg.det(Vt.T @ U.T)
    D  = np.diag([1, 1, d])
    R  = Vt.T @ D @ U.T

    P_rot = P @ R.T
    rmsd  = float(np.sqrt(((P_rot - Q) ** 2).sum(-1).mean()))
    return rmsd, P_rot


def compute_tm_score(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Compute TM-score between predicted and true C1' coordinates.
    Both: (L, 3) numpy arrays.
    """
    L = len(true)
    if L < 2:
        return 0.0

    d0 = max(1.24 * (L - 15) ** (1/3) - 1.8, 0.5) if L > 21 else 0.5

    # superpose
    rmsd, pred_rot = kabsch_rmsd(pred, true)

    di2 = ((pred_rot - true) ** 2).sum(-1)
    tm  = (1.0 / (1.0 + di2 / d0**2)).mean()
    return float(tm)


def compute_gdt_ts(pred: np.ndarray, true: np.ndarray,
                   cutoffs=(1, 2, 4, 8)) -> float:
    """GDT_TS: fraction of residues within 1/2/4/8 Å after superposition."""
    _, pred_rot = kabsch_rmsd(pred, true)
    di = np.sqrt(((pred_rot - true) ** 2).sum(-1))
    fracs = [( di <= c).mean() for c in cutoffs]
    return float(np.mean(fracs))


def evaluate_predictions(pred_df: pd.DataFrame,
                          true_df: pd.DataFrame,
                          id_col: str = 'target_id') -> pd.DataFrame:
    """
    Evaluate predictions vs ground-truth DataFrame.
    Both DataFrames must have columns: target_id, resid, x_1, y_1, z_1
    Returns per-target metrics DataFrame.
    """
    results = []
    for tid in pred_df[id_col].unique():
        p = pred_df[pred_df[id_col] == tid].sort_values('resid')
        t = true_df[true_df[id_col] == tid].sort_values('resid')

        common = set(p['resid']) & set(t['resid'])
        p = p[p['resid'].isin(common)].sort_values('resid')
        t = t[t['resid'].isin(common)].sort_values('resid')

        if len(p) < 3:
            continue

        pc = p[['x_1', 'y_1', 'z_1']].values.astype(np.float32)
        tc = t[['x_1', 'y_1', 'z_1']].values.astype(np.float32)

        # drop NaN
        valid = ~(np.isnan(pc).any(1) | np.isnan(tc).any(1))
        pc, tc = pc[valid], tc[valid]
        if len(pc) < 3:
            continue

        rmsd, _ = kabsch_rmsd(pc, tc)
        tm       = compute_tm_score(pc, tc)
        gdt      = compute_gdt_ts(pc, tc)

        results.append({'target_id': tid, 'n_res': len(pc),
                         'RMSD': rmsd, 'TM_score': tm, 'GDT_TS': gdt})

    df = pd.DataFrame(results)
    if len(df):
        print("\n=== EVALUATION SUMMARY ===")
        print(f"  Targets   : {len(df)}")
        print(f"  Mean RMSD : {df['RMSD'].mean():.3f} Å")
        print(f"  Mean TM   : {df['TM_score'].mean():.4f}")
        print(f"  Mean GDT  : {df['GDT_TS'].mean():.4f}")
    return df


# ─────────────────────────────────────────────────────────────
# 7. ENSEMBLE AVERAGING
# ─────────────────────────────────────────────────────────────
def ensemble_predictions(pred_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Average coordinates from multiple prediction DataFrames.
    All must have same target_ids and resids.
    """
    assert len(pred_list) > 0
    if len(pred_list) == 1:
        return pred_list[0]

    base = pred_list[0][['target_id', 'resid']].copy()
    for col in ['x_1', 'y_1', 'z_1']:
        base[col] = np.mean([df[col].values for df in pred_list], axis=0)
    return base


# ─────────────────────────────────────────────────────────────
# 8. POST-PROCESS PREDICTIONS & FORMAT SUBMISSION
# ─────────────────────────────────────────────────────────────
def post_process_predictions(pred_df: pd.DataFrame,
                             apply_physics: bool = True) -> pd.DataFrame:
    """
    Apply physics refinement to all predicted structures in DataFrame.
    DataFrame columns: target_id, resid, x_1, y_1, z_1
    """
    if not apply_physics:
        return pred_df

    refined_rows = []
    for tid, grp in tqdm(pred_df.groupby('target_id'), desc="Refining"):
        grp = grp.sort_values('resid')
        coords = grp[['x_1', 'y_1', 'z_1']].values.astype(np.float32)

        coords_refined = refine_structure(coords, use_gradient=False)  # fast mode

        for i, row in enumerate(grp.itertuples(index=False)):
            refined_rows.append({
                'target_id': tid,
                'resid'    : row.resid,
                'x_1'      : float(coords_refined[i, 0]),
                'y_1'      : float(coords_refined[i, 1]),
                'z_1'      : float(coords_refined[i, 2]),
            })

    return pd.DataFrame(refined_rows)


def format_submission(pred_df: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """
    Format final submission CSV.
    Adds ID column in format: {target_id}_{resid}
    """
    df = pred_df.copy()
    df['ID'] = df['target_id'] + '_' + df['resid'].astype(str)
    df = df[['ID', 'x_1', 'y_1', 'z_1']].reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}  ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────────────────────
# 9. MAIN POST-PROCESSING RUNNER
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    OUT_DIR = "/kaggle/working"
    VALID_CSV = "/kaggle/input/competitions/stanford-rna-3d-folding-2/validation_sequences.csv"

    # ── Load raw predictions (from main pipeline) ──
    val_pred_path = os.path.join(OUT_DIR, "predictions_validation.csv")
    test_pred_path = os.path.join(OUT_DIR, "predictions_test.csv")

    if Path(val_pred_path).exists():
        print("Post-processing validation predictions...")
        val_pred = pd.read_csv(val_pred_path)
        val_refined = post_process_predictions(val_pred, apply_physics=True)
        val_refined.to_csv(os.path.join(OUT_DIR, "validation_refined.csv"), index=False)

        # evaluate if ground truth available
        # (you would load true coords here from PDB)
        print(f"Refined {val_pred['target_id'].nunique()} validation structures")

    if Path(test_pred_path).exists():
        print("\nPost-processing test predictions...")
        test_pred = pd.read_csv(test_pred_path)
        test_refined = post_process_predictions(test_pred, apply_physics=True)
        sub = format_submission(test_refined, os.path.join(OUT_DIR, "submission.csv"))
        print(sub.head())

    print("\n✅ Post-processing complete!")
