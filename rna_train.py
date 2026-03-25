"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TRAINING PIPELINE — KAGGLE WINNING LEVEL                                   ║
║  Dataset · Multi-task Loss · Training Loop · Inference · Submission         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, gc, math, random, warnings
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
from torch.cuda.amp import GradScaler, autocast
from Bio.PDB import MMCIFParser

# local modules
from rna_features import build_all_features, geometric_features
from rna_model_se3 import RNAFoldSE3, cfg as ModelCfg

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS & TRAINING CONFIG
# ─────────────────────────────────────────────────────────────
class TrainConfig:
    BASE      = "/kaggle/input/competitions/stanford-rna-3d-folding-2"
    MSA_DIR   = f"{BASE}/MSA"
    PDB_DIR   = f"{BASE}/PDB_RNA"
    TRAIN_CSV = f"{BASE}/train_sequences.csv"
    VALID_CSV = f"{BASE}/validation_sequences.csv"
    TEST_CSV  = f"{BASE}/test_sequences.csv"
    OUT_DIR   = "/kaggle/working"

    BATCH_SIZE   = 2
    EPOCHS       = 40
    LR           = 1e-4
    LR_PAIR      = 3e-4   # higher lr for pair embedding
    WEIGHT_DECAY = 0.01
    GRAD_CLIP    = 1.0
    WARMUP_STEPS = 1000
    MAX_LEN      = ModelCfg.MAX_LEN

    # Loss weights
    W_COORD    = 1.0
    W_TM       = 0.5
    W_DIST     = 0.3     # distogram cross-entropy
    W_CONTACT  = 0.2
    W_RECYCLE  = 0.3     # weight for intermediate recycles
    LABEL_SMOOTH = 0.1

    SEED         = 42
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS  = 2
    MIXED_PREC   = True

cfg = TrainConfig()

RNA_MAP = {
    'A':'A','U':'U','G':'G','C':'C',
    'RA':'A','RU':'U','RG':'G','RC':'C',
    '2MG':'G','1MG':'G','7MG':'G','5MC':'C',
    'H2U':'U','PSU':'U','I':'G','M2G':'G',
}
VOCAB = {'A':0,'U':1,'G':2,'C':3,'<PAD>':4,'<UNK>':5}
_cif_parser = MMCIFParser(QUIET=True)

def seed_all(seed=cfg.SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

seed_all()

# ─────────────────────────────────────────────────────────────
# CIF COORDINATE LOADER
# ─────────────────────────────────────────────────────────────
def load_cif_coords(target_id: str) -> Optional[Tuple[str, np.ndarray]]:
    cif = Path(cfg.PDB_DIR) / f"{target_id.lower()}.cif"
    if not cif.exists(): return None
    try:
        structure = _cif_parser.get_structure(target_id, str(cif))
        model = structure[0]
        seq, coords = [], []
        for chain in model:
            for res in chain:
                rn = res.resname.strip()
                base = RNA_MAP.get(rn, RNA_MAP.get(rn[-1] if rn else 'X', None))
                if base is None: continue
                seq.append(base)
                coords.append(res["C1'"].coord if "C1'" in res else [np.nan]*3)
        if not seq: return None
        return ''.join(seq), np.array(coords, np.float32)
    except:
        return None


def align_coords(csv_seq, cif_seq, cif_coords, max_len):
    L = min(len(csv_seq), max_len)
    csv_seq_t = csv_seq[:L]
    idx = cif_seq.find(csv_seq_t)
    if idx >= 0:
        c = cif_coords[idx:idx+L]
    else:
        # fallback: first L
        c = cif_coords[:L]
    if len(c) < L:
        pad = np.full((L - len(c), 3), np.nan, np.float32)
        c = np.concatenate([c, pad])
    return c


# ─────────────────────────────────────────────────────────────
# DISTANCE BINS (for distogram target)
# ─────────────────────────────────────────────────────────────
D_MIN, D_MAX, N_BINS = 2.0, 20.0, 36
BIN_EDGES = np.linspace(D_MIN, D_MAX, N_BINS + 1)

def coords_to_dist_bins(coords: np.ndarray, L: int) -> np.ndarray:
    """(L_raw, 3) → (L, L) binned index array (int64)."""
    Lr = min(len(coords), L)
    dist = np.zeros((L, L), np.float32)
    valid = ~np.isnan(coords[:Lr, 0])
    idx = np.where(valid)[0]
    for ii, i in enumerate(idx):
        for jj, j in enumerate(idx[ii+1:], ii+1):
            d = np.linalg.norm(coords[i] - coords[j])
            dist[i, j] = dist[j, i] = d
    bins = np.digitize(dist, BIN_EDGES) - 1
    bins = np.clip(bins, 0, N_BINS - 1).astype(np.int64)
    return bins  # (L, L)


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class RNADataset(Dataset):
    def __init__(self, csv_path: str, is_test: bool = False):
        self.df      = pd.read_csv(csv_path)
        self.is_test = is_test
        self.L       = cfg.MAX_LEN

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        tid       = row['target_id']
        seq       = row['sequence']
        L_seq     = min(len(seq), self.L)
        L         = self.L

        # ── Token IDs ──
        seq_ids = torch.tensor([VOCAB.get(c, VOCAB['<UNK>']) for c in seq[:L]]
                               + [VOCAB['<PAD>']] * (L - L_seq), dtype=torch.long)
        seq_mask = torch.zeros(L, dtype=torch.bool)
        seq_mask[:L_seq] = True

        # ── Load coords (if available) ──
        coords_gt   = np.full((L, 3), np.nan, np.float32)
        coord_mask  = np.zeros(L, np.float32)
        dist_bins_t = np.zeros((L, L), np.int64)

        if not self.is_test:
            result = load_cif_coords(tid)
            if result is not None:
                cif_seq, cif_coords = result
                aligned = align_coords(seq, cif_seq, cif_coords, L)
                valid   = ~np.isnan(aligned[:, 0])
                coords_gt[valid]  = aligned[valid]
                coord_mask[valid] = 1.0
                dist_bins_t       = coords_to_dist_bins(aligned, L)

        # ── Build all features ──
        feats = build_all_features(
            seq=seq, target_id=tid,
            coords=(coords_gt if coord_mask.sum() > 0 else None),
            msa_dir=cfg.MSA_DIR,
            max_len=L,
        )

        # ── Contact target (from true dist) ──
        contact_t = np.zeros((L, L), np.float32)
        if coord_mask.sum() > 2:
            dist_full = np.zeros((L, L), np.float32)
            vi = np.where(coord_mask > 0)[0]
            for ii, i in enumerate(vi):
                for j in vi[ii+1:]:
                    d = np.linalg.norm(coords_gt[i] - coords_gt[j])
                    dist_full[i, j] = dist_full[j, i] = d
            contact_t = (dist_full < 8.0).astype(np.float32)
            np.fill_diagonal(contact_t, 0)

        return {
            # inputs
            'seq_ids'  : seq_ids,
            'seq_mask' : seq_mask,
            'f1'       : torch.from_numpy(feats['f1'][:L]),             # (L, 5)
            'dihed'    : torch.from_numpy(feats['dihed']),               # (L, 4)
            'dist_rbf' : torch.from_numpy(feats['dist_rbf']),            # (L,L,16)
            'dist_bins': torch.from_numpy(feats['dist_bins']),           # (L,L,36)
            'orient'   : torch.from_numpy(feats['orient']),              # (L,L,4)
            'rel_pos'  : torch.from_numpy(feats['rel_pos']),             # (L,L,65)
            'MIp'      : torch.from_numpy(feats['MIp']),                 # (L,L)
            'FNp'      : torch.from_numpy(feats['FNp']),                 # (L,L)
            # targets
            'coords'    : torch.from_numpy(coords_gt),                   # (L,3)
            'coord_mask': torch.from_numpy(coord_mask),                  # (L,)
            'dist_bins_t': torch.from_numpy(dist_bins_t),                # (L,L)
            'contact_t' : torch.from_numpy(contact_t),                   # (L,L)
            'seq_len'   : L_seq,
            'target_id' : tid,
        }


def collate_fn(batch):
    scalar_keys = ['seq_len', 'target_id']
    tensor_keys = [k for k in batch[0] if k not in scalar_keys]
    out = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    out['seq_len']   = [b['seq_len'] for b in batch]
    out['target_id'] = [b['target_id'] for b in batch]
    return out


# ─────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────
def tm_loss(pred, true, mask, seq_len):
    """
    Differentiable TM-score loss.
    pred, true: (B, L, 3); mask: (B, L); seq_len: list[int]
    """
    B = pred.shape[0]
    loss = pred.new_zeros(1)
    for b in range(B):
        L  = seq_len[b]
        d0 = max(1.24 * (L - 15) ** (1/3) - 1.8, 0.5) if L > 21 else 0.5
        m  = mask[b, :L].bool()
        if m.sum() < 3: continue
        p  = pred[b, :L][m] - pred[b, :L][m].mean(0)
        t  = true[b, :L][m] - true[b, :L][m].mean(0)
        di2 = ((p - t)**2).sum(-1)
        tm  = (1.0 / (1.0 + di2 / d0**2)).mean()
        loss = loss - tm   # minimize negative TM
    return loss / B


def frame_aligned_point_error(pred, true, mask, clamp_dist=10.0):
    """
    FAPE loss (AlphaFold2-style).
    Computes pairwise frame-aligned distances, clamps, and averages.
    pred, true: (B, L, 3); mask: (B, L)
    """
    B, L, _ = pred.shape
    loss = pred.new_zeros(1)
    for b in range(B):
        m   = mask[b].bool()
        if m.sum() < 3: continue
        p   = pred[b][m]   # (N, 3)
        t   = true[b][m]
        # all pairwise differences
        dp  = p.unsqueeze(0) - p.unsqueeze(1)   # (N, N, 3)
        dt  = t.unsqueeze(0) - t.unsqueeze(1)
        err = ((dp - dt)**2).sum(-1).clamp(max=clamp_dist**2).sqrt()
        loss = loss + err.mean()
    return loss / B


def distogram_loss(pred_bins, true_bins, mask, label_smooth=cfg.LABEL_SMOOTH):
    """
    Cross-entropy on distogram bins with label smoothing.
    pred_bins : (B, L, L, n_bins)
    true_bins : (B, L, L) long
    mask      : (B, L)
    """
    B, L, _, K = pred_bins.shape
    pair_mask = mask.unsqueeze(2).float() * mask.unsqueeze(1).float()  # (B,L,L)

    pred_flat = pred_bins.reshape(-1, K)
    true_flat = true_bins.reshape(-1).to(pred_bins.device)
    m_flat    = pair_mask.reshape(-1).bool()

    if m_flat.sum() == 0:
        return pred_bins.new_zeros(1)

    ce = F.cross_entropy(pred_flat[m_flat], true_flat[m_flat],
                         label_smoothing=label_smooth)
    return ce


def contact_loss_fn(pred, true, mask):
    """Weighted BCE for contact prediction."""
    pair_mask = mask.unsqueeze(2).float() * mask.unsqueeze(1).float()
    true = true.to(pred.device)
    # positive weight to handle imbalance
    pos_weight = torch.tensor(5.0, device=pred.device)
    loss = F.binary_cross_entropy_with_logits(
        torch.logit(pred.clamp(1e-6, 1-1e-6)),
        true,
        pos_weight=pos_weight,
        reduction='none',
    )
    return (loss * pair_mask).sum() / (pair_mask.sum() + 1e-8)


def multi_task_loss(outputs, batch):
    pred_coords = outputs['coords']
    all_coords  = outputs['all_coords']
    pred_bins   = outputs['distogram']
    pred_cont   = outputs['contact']

    true_coords = batch['coords'].to(cfg.DEVICE)
    coord_mask  = batch['coord_mask'].to(cfg.DEVICE)
    true_bins   = batch['dist_bins_t'].to(cfg.DEVICE)
    true_cont   = batch['contact_t'].to(cfg.DEVICE)
    seq_mask    = batch['seq_mask'].to(cfg.DEVICE)
    seq_len     = batch['seq_len']

    # primary coordinate losses
    # MSE
    m = coord_mask.bool().unsqueeze(-1).expand_as(pred_coords)
    l_mse  = F.mse_loss(pred_coords[m], true_coords[m]) if m.any() else pred_coords.new_zeros(1)

    # TM-score
    l_tm   = tm_loss(pred_coords, true_coords, coord_mask, seq_len)

    # FAPE
    l_fape = frame_aligned_point_error(pred_coords, true_coords, coord_mask)

    # distogram
    l_dist = distogram_loss(pred_bins, true_bins, seq_mask)

    # contact
    l_cont = contact_loss_fn(pred_cont, true_cont, seq_mask)

    # auxiliary recycle losses (progressively weighted)
    l_recycle = pred_coords.new_zeros(1)
    for i, coords_i in enumerate(all_coords[:-1]):
        w = cfg.W_RECYCLE * (i + 1) / len(all_coords)
        mi = coord_mask.bool().unsqueeze(-1).expand_as(coords_i)
        if mi.any():
            l_recycle = l_recycle + w * F.mse_loss(coords_i[mi], true_coords[mi])

    total = (cfg.W_COORD * (l_mse + l_fape) +
             cfg.W_TM    * l_tm +
             cfg.W_DIST  * l_dist +
             cfg.W_CONTACT * l_cont +
             l_recycle)

    parts = {
        'mse'    : l_mse.item(),
        'tm'     : l_tm.item(),
        'fape'   : l_fape.item(),
        'dist'   : l_dist.item(),
        'contact': l_cont.item(),
        'recycle': l_recycle.item(),
    }
    return total, parts


# ─────────────────────────────────────────────────────────────
# WARMUP COSINE SCHEDULER
# ─────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.opt         = optimizer
        self.warmup      = warmup_steps
        self.total       = total_steps
        self.min_lr      = min_lr
        self.base_lrs    = [g['lr'] for g in optimizer.param_groups]
        self._step       = 0

    def step(self):
        self._step += 1
        s = self._step
        for i, g in enumerate(self.opt.param_groups):
            base = self.base_lrs[i]
            if s < self.warmup:
                lr = base * s / self.warmup
            else:
                progress = (s - self.warmup) / max(self.total - self.warmup, 1)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (1 + math.cos(math.pi * progress))
            g['lr'] = lr

    def get_last_lr(self):
        return [g['lr'] for g in self.opt.param_groups]


# ─────────────────────────────────────────────────────────────
# TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler, epoch):
    model.train()
    total, n = 0.0, 0
    agg = {k: 0.0 for k in ['mse','tm','fape','dist','contact','recycle']}

    bar = tqdm(loader, desc=f"Epoch {epoch:02d}", leave=False, dynamic_ncols=True)
    for batch in bar:
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.MIXED_PREC):
            outputs = model(batch, device=cfg.DEVICE)
            loss, parts = multi_task_loss(outputs, batch)

        if cfg.MIXED_PREC:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

        scheduler.step()

        total += loss.item(); n += 1
        for k in agg: agg[k] += parts[k]
        bar.set_postfix(loss=f"{total/n:.4f}", tm=f"{agg['tm']/n:.3f}")

    return total / n, {k: v / n for k, v in agg.items()}


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="  Val", leave=False, dynamic_ncols=True):
        with autocast(enabled=cfg.MIXED_PREC):
            outputs = model(batch, device=cfg.DEVICE)
            loss, _ = multi_task_loss(outputs, batch)
        total += loss.item(); n += 1
    return total / n


# ─────────────────────────────────────────────────────────────
# TRAINING ENTRY POINT
# ─────────────────────────────────────────────────────────────
def run_training():
    print(f"\n{'='*70}")
    print(f"  RNA 3D FOLDING — SE(3) Equivariant Pipeline")
    print(f"  Device: {cfg.DEVICE}")
    print(f"{'='*70}")

    train_ds = RNADataset(cfg.TRAIN_CSV, is_test=False)
    valid_ds = RNADataset(cfg.VALID_CSV, is_test=False)
    print(f"Train: {len(train_ds)} | Valid: {len(valid_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS,
        pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS,
    )

    model = RNAFoldSE3().to(cfg.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # differential learning rates
    pair_params   = [p for n, p in model.named_parameters() if 'pair_embed' in n]
    other_params  = [p for n, p in model.named_parameters() if 'pair_embed' not in n]
    optimizer = AdamW([
        {'params': pair_params,  'lr': cfg.LR_PAIR},
        {'params': other_params, 'lr': cfg.LR},
    ], weight_decay=cfg.WEIGHT_DECAY)

    total_steps = cfg.EPOCHS * len(train_loader)
    scheduler   = WarmupCosineScheduler(optimizer, cfg.WARMUP_STEPS, total_steps)
    scaler      = GradScaler(enabled=cfg.MIXED_PREC)

    best_val = float('inf')
    history  = []
    ckpt_path = os.path.join(cfg.OUT_DIR, 'best_rna_se3.pt')

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, parts = train_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)
        val_loss          = eval_epoch(model, valid_loader)

        lrs = scheduler.get_last_lr()
        print(f"Ep {epoch:3d}/{cfg.EPOCHS} | "
              f"train={train_loss:.4f}  "
              f"(mse={parts['mse']:.3f} tm={parts['tm']:.3f} "
              f"fape={parts['fape']:.3f} dist={parts['dist']:.3f}) | "
              f"val={val_loss:.4f} | lr={lrs[0]:.2e}")

        history.append({'epoch': epoch, 'train': train_loss, 'val': val_loss, **parts})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_loss': val_loss, 'history': history}, ckpt_path)
            print(f"   ✅  Best saved (val={best_val:.4f})")

        gc.collect()
        if cfg.DEVICE == 'cuda': torch.cuda.empty_cache()

    # save history
    pd.DataFrame(history).to_csv(
        os.path.join(cfg.OUT_DIR, 'training_history.csv'), index=False)
    return model, history


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_dataset(model, csv_path: str, is_test: bool) -> pd.DataFrame:
    model.eval()
    ds     = RNADataset(csv_path, is_test=is_test)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS)
    rows = []
    for batch in tqdm(loader, desc=f"  Predicting {Path(csv_path).stem}"):
        with autocast(enabled=cfg.MIXED_PREC):
            outputs = model(batch, device=cfg.DEVICE)
        coords = outputs['coords'].cpu().float().numpy()  # (B, L, 3)
        for b, tid in enumerate(batch['target_id']):
            L = batch['seq_len'][b]
            for i in range(L):
                rows.append({'target_id': tid, 'resid': i+1,
                             'x_1': float(coords[b, i, 0]),
                             'y_1': float(coords[b, i, 1]),
                             'z_1': float(coords[b, i, 2])})
    return pd.DataFrame(rows)


def run_inference(model=None):
    if model is None:
        ckpt = os.path.join(cfg.OUT_DIR, 'best_rna_se3.pt')
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt}")
        model = RNAFoldSE3().to(cfg.DEVICE)
        state = torch.load(ckpt, map_location=cfg.DEVICE)
        model.load_state_dict(state['model'])
        print(f"Loaded checkpoint  epoch={state['epoch']}  val={state['val_loss']:.4f}")

    for split, cpath, is_t in [
        ('validation', cfg.VALID_CSV, False),
        ('test',       cfg.TEST_CSV,  True),
    ]:
        if not Path(cpath).exists():
            print(f"[skip] {cpath} not found"); continue
        df = predict_dataset(model, cpath, is_t)
        out = os.path.join(cfg.OUT_DIR, f'predictions_{split}.csv')
        df.to_csv(out, index=False)
        print(f"Saved {split}: {out}  ({len(df)} rows)")


# ─────────────────────────────────────────────────────────────
# POST-PROCESSING & SUBMISSION
# ─────────────────────────────────────────────────────────────
def refine_and_submit(pred_path: str, out_path: str):
    """
    Apply physics-based refinement (from rna_physics_refinement.py)
    then write submission CSV.
    """
    try:
        from rna_physics_refinement import post_process_predictions, format_submission
        df  = pd.read_csv(pred_path)
        df  = post_process_predictions(df, apply_physics=True)
        sub = format_submission(df, out_path)
        return sub
    except ImportError:
        # fallback: just rename columns
        df  = pd.read_csv(pred_path)
        df['ID'] = df['target_id'] + '_' + df['resid'].astype(str)
        sub = df[['ID', 'x_1', 'y_1', 'z_1']]
        sub.to_csv(out_path, index=False)
        print(f"Submission saved → {out_path}")
        return sub


# ─────────────────────────────────────────────────────────────
# EVALUATION (TM-score on validation)
# ─────────────────────────────────────────────────────────────
def evaluate_on_validation():
    """
    Compute TM-score / RMSD between predictions and true PDB coords
    for validation set.
    """
    pred_path = os.path.join(cfg.OUT_DIR, 'predictions_validation.csv')
    if not Path(pred_path).exists():
        print("No validation predictions found."); return

    pred_df = pd.read_csv(pred_path)
    valid_df = pd.read_csv(cfg.VALID_CSV)
    results  = []

    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Evaluating"):
        tid = row['target_id']
        seq = row['sequence']
        result = load_cif_coords(tid)
        if result is None: continue

        cif_seq, cif_coords = result
        aligned = align_coords(seq, cif_seq, cif_coords, cfg.MAX_LEN)
        valid   = ~np.isnan(aligned[:, 0])
        if valid.sum() < 3: continue

        true_c = aligned[valid]

        pred_grp = pred_df[pred_df['target_id'] == tid].sort_values('resid')
        if len(pred_grp) == 0: continue
        pred_c = pred_grp[['x_1', 'y_1', 'z_1']].values.astype(np.float32)

        # align lengths
        mn = min(len(true_c), len(pred_c))
        true_c, pred_c = true_c[:mn], pred_c[:mn]

        # Kabsch superposition
        def kabsch(P, Q):
            P = P - P.mean(0); Q = Q - Q.mean(0)
            H = P.T @ Q
            U, S, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            D = np.diag([1, 1, d])
            R = Vt.T @ D @ U.T
            return P @ R.T, Q

        P_rot, Q = kabsch(pred_c, true_c)
        rmsd = float(np.sqrt(((P_rot - Q)**2).sum(-1).mean()))

        L = mn
        d0 = max(1.24*(L-15)**(1/3)-1.8, 0.5) if L > 21 else 0.5
        di2 = ((P_rot - Q)**2).sum(-1)
        tm  = float((1/(1+di2/d0**2)).mean())

        results.append({'target_id': tid, 'L': L, 'RMSD': rmsd, 'TM': tm})

    df = pd.DataFrame(results)
    if len(df):
        print(f"\n{'='*50}")
        print(f"  Validation Results ({len(df)} targets)")
        print(f"  Mean RMSD : {df['RMSD'].mean():.3f} Å ± {df['RMSD'].std():.3f}")
        print(f"  Mean TM   : {df['TM'].mean():.4f} ± {df['TM'].std():.4f}")
        print(f"  TM > 0.5  : {(df['TM'] > 0.5).mean()*100:.1f}%")
        print(f"{'='*50}")
        df.to_csv(os.path.join(cfg.OUT_DIR, 'validation_metrics.csv'), index=False)
    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode in ("train", "full"):
        model, history = run_training()
    else:
        model = None

    if mode in ("infer", "full"):
        run_inference(model)

    if mode in ("eval", "full"):
        evaluate_on_validation()

    if mode in ("submit", "full"):
        test_pred = os.path.join(cfg.OUT_DIR, 'predictions_test.csv')
        sub_path  = os.path.join(cfg.OUT_DIR, 'submission.csv')
        if Path(test_pred).exists():
            refine_and_submit(test_pred, sub_path)
            print(pd.read_csv(sub_path).head())

    print("\n🎉 Done!")
