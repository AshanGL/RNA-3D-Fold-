"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RNA 3D FOLDING — TRAINING PIPELINE v3                                       ║
║                                                                              ║
║  KEY SPEEDUP vs v2:                                                          ║
║   • All features (MSA covariation, Nussinov, geometry, RBF, rel-pos) are    ║
║     computed ONCE before training and saved to /kaggle/working/feat_cache/  ║
║   • __getitem__ is now pure file-read + torch.from_numpy — ~50-200× faster ║
║     than recomputing features on every step.                                 ║
║   • GPU (T4) used for batched pairwise distance / RBF during caching.       ║
║   • DataLoader runs with num_workers=4 and pin_memory=True so GPU is fed    ║
║     while it processes the previous batch (overlapping I/O with compute).   ║
║   • Mixed-precision (AMP) keeps T4 Tensor Cores busy.                       ║
║                                                                              ║
║  Everything else (model, losses, evaluation, submission) is unchanged.      ║
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
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from Bio.PDB import MMCIFParser

# ── local modules ──
from rna_features_v2 import find_msa_file, chunk_sequence, stitch_coords
from rna_model_se3_v2 import RNAFoldSE3, build_model_dual_gpu, cfg as ModelCfg
from rna_feature_cache import (
    CachedRNADataset, collate_fn,
    precompute_split, compute_and_save_features,
    gpu_distance_features,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────────────────────
class TrainConfig:
    # Paths
    BASE      = "/kaggle/input/competitions/stanford-rna-3d-folding-2"
    MSA_DIR   = f"{BASE}/MSA"
    PDB_DIR   = f"{BASE}/PDB_RNA"
    TRAIN_CSV = f"{BASE}/train_sequences.csv"
    VALID_CSV = f"{BASE}/validation_sequences.csv"
    TEST_CSV  = f"{BASE}/test_sequences.csv"
    OUT_DIR   = "/kaggle/working"
    # Feature cache directory — persists across runs (re-used if not stale)
    CACHE_DIR = "/kaggle/working/feat_cache"

    # ── DATA ─────────────────────────────────────────────────
    TRAIN_FRAC   = 1.0
    MAX_LEN      = ModelCfg.MAX_LEN   # 512

    # ── MODEL ────────────────────────────────────────────────
    BATCH_SIZE   = 4          # per GPU; with 2 GPUs effective = 8
    EPOCHS       = 40
    LR           = 1e-4
    LR_PAIR      = 3e-4
    WEIGHT_DECAY = 0.01
    GRAD_CLIP    = 1.0
    WARMUP_STEPS = 1000
    MIXED_PREC   = True

    # ── LOSS WEIGHTS ─────────────────────────────────────────
    W_COORD    = 1.0
    W_TM       = 0.5
    W_FAPE     = 0.5
    W_DIST     = 0.3
    W_CONTACT  = 0.2
    W_RECYCLE  = 0.3
    LABEL_SMOOTH = 0.1

    # ── DATALOADER ───────────────────────────────────────────
    # With pre-cached features workers can safely read .npz files in parallel.
    # 4 workers fills the GPU pipeline without OOM on Kaggle T4 × 2.
    NUM_WORKERS  = 4
    PIN_MEMORY   = True       # DMA-pinned host memory → faster H2D transfer

    # ── MISC ─────────────────────────────────────────────────
    SEED = 42

cfg = TrainConfig()

# ─────────────────────────────────────────────────────────────
# CONSTANTS (shared with v2)
# ─────────────────────────────────────────────────────────────
RNA_MAP = {
    'A':'A','U':'U','G':'G','C':'C',
    'RA':'A','RU':'U','RG':'G','RC':'C',
    '2MG':'G','1MG':'G','7MG':'G','5MC':'C',
    'H2U':'U','PSU':'U','I':'G','M2G':'G',
}
VOCAB    = {'A':0,'U':1,'G':2,'C':3,'<PAD>':4,'<UNK>':5}
D_MIN, D_MAX, N_BINS = 2.0, 20.0, 36
BIN_EDGES = np.linspace(D_MIN, D_MAX, N_BINS + 1)

_cif_parser = MMCIFParser(QUIET=True)


def seed_all(seed=cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_all()


# ─────────────────────────────────────────────────────────────
# CIF COORDINATE LOADER  (unchanged from v2)
# ─────────────────────────────────────────────────────────────
def find_cif_file(target_id: str) -> Optional[Path]:
    pdb_dir = Path(cfg.PDB_DIR)
    for name in [f"{target_id.lower()}.cif",
                 f"{target_id.upper()}.cif",
                 f"{target_id}.cif"]:
        p = pdb_dir / name
        if p.exists():
            return p
    return None


def load_cif_coords(target_id: str) -> Optional[Tuple[str, np.ndarray]]:
    p = find_cif_file(target_id)
    if p is None:
        return None
    try:
        structure = _cif_parser.get_structure(target_id, str(p))
        model     = structure[0]
        seq, coords = [], []
        for chain in model:
            for res in chain:
                rn   = res.resname.strip()
                base = RNA_MAP.get(rn, RNA_MAP.get(rn[-1] if rn else 'X', None))
                if base is None:
                    continue
                seq.append(base)
                coords.append(res["C1'"].coord if "C1'" in res else [np.nan]*3)
        if not seq:
            return None
        return ''.join(seq), np.array(coords, np.float32)
    except Exception as e:
        print(f"  [CIF warn] {target_id}: {e}")
        return None


def align_coords(csv_seq: str, cif_seq: str,
                 cif_coords: np.ndarray, max_len: int) -> np.ndarray:
    L   = min(len(csv_seq), max_len)
    seq = csv_seq[:L]
    idx = cif_seq.find(seq)
    if idx >= 0:
        c = cif_coords[idx:idx+L]
    else:
        c = cif_coords[:L]
    if len(c) < L:
        pad = np.full((L - len(c), 3), np.nan, np.float32)
        c   = np.concatenate([c, pad])
    return c


# ─────────────────────────────────────────────────────────────
# ROW-DICT BUILDER  (used by precompute_split)
# ─────────────────────────────────────────────────────────────
def _build_row_dicts(df: pd.DataFrame, split: str) -> List[dict]:
    """
    For each row in df, load CIF coords (if available) and build a row_dict
    ready for compute_and_save_features().
    """
    rows = []
    for _, row in df.iterrows():
        tid = row['target_id']
        seq = str(row['sequence'])
        coords = None
        if split != 'test':
            result = load_cif_coords(tid)
            if result is not None:
                cif_seq, cif_coords = result
                coords = align_coords(seq, cif_seq, cif_coords, cfg.MAX_LEN)
        rows.append({
            'target_id': tid,
            'sequence' : seq,
            'split'    : split,
            'coords'   : coords,
        })
    return rows


# ─────────────────────────────────────────────────────────────
# LOSS FUNCTIONS  (identical to v2)
# ─────────────────────────────────────────────────────────────
def tm_loss(pred, true, mask, seq_len):
    B    = pred.shape[0]
    loss = pred.new_zeros(1)
    for b in range(B):
        L  = seq_len[b]
        d0 = max(1.24 * (L - 15) ** (1/3) - 1.8, 0.5) if L > 21 else 0.5
        m  = mask[b, :L].bool()
        if m.sum() < 3:
            continue
        p   = pred[b, :L][m] - pred[b, :L][m].mean(0)
        t   = true[b, :L][m] - true[b, :L][m].mean(0)
        di2 = ((p - t)**2).sum(-1)
        tm  = (1.0 / (1.0 + di2 / d0**2)).mean()
        loss = loss - tm
    return loss / B


def fape_loss(pred, true, mask, clamp=10.0):
    B, L, _ = pred.shape
    loss = pred.new_zeros(1)
    for b in range(B):
        m = mask[b].bool()
        if m.sum() < 3:
            continue
        p  = pred[b][m]
        t  = true[b][m]
        dp = p.unsqueeze(0) - p.unsqueeze(1)
        dt = t.unsqueeze(0) - t.unsqueeze(1)
        err = ((dp - dt)**2).sum(-1).clamp(max=clamp**2).sqrt()
        loss = loss + err.mean()
    return loss / B


def distogram_loss(pred_bins, true_bins, mask, smooth=cfg.LABEL_SMOOTH):
    B, L, _, K = pred_bins.shape
    pm   = mask.unsqueeze(2).float() * mask.unsqueeze(1).float()
    pf   = pred_bins.reshape(-1, K)
    tf   = true_bins.reshape(-1).to(pred_bins.device)
    mf   = pm.reshape(-1).bool()
    if mf.sum() == 0:
        return pred_bins.new_zeros(1)
    return F.cross_entropy(pf[mf], tf[mf], label_smoothing=smooth)


def contact_loss_fn(pred, true, mask):
    pm   = mask.unsqueeze(2).float() * mask.unsqueeze(1).float()
    true = true.to(pred.device)
    pw   = torch.tensor(5.0, device=pred.device)
    loss = F.binary_cross_entropy_with_logits(
        torch.logit(pred.clamp(1e-6, 1-1e-6)), true,
        pos_weight=pw, reduction='none',
    )
    return (loss * pm).sum() / (pm.sum() + 1e-8)


def multi_task_loss(outputs, batch, device):
    pred_coords = outputs['coords']
    all_coords  = outputs['all_coords']
    pred_bins   = outputs['distogram']
    pred_cont   = outputs['contact']

    true_coords = batch['coords'].to(device)
    coord_mask  = batch['coord_mask'].to(device)
    true_bins   = batch['dist_bins_t'].to(device)
    true_cont   = batch['contact_t'].to(device)
    seq_mask    = batch['seq_mask'].to(device)
    seq_len     = batch['seq_len']

    m      = coord_mask.bool().unsqueeze(-1).expand_as(pred_coords)
    l_mse  = F.mse_loss(pred_coords[m], true_coords[m]) if m.any() else pred_coords.new_zeros(1)
    l_tm   = tm_loss(pred_coords, true_coords, coord_mask, seq_len)
    l_fape = fape_loss(pred_coords, true_coords, coord_mask)
    l_dist = distogram_loss(pred_bins, true_bins, seq_mask)
    l_cont = contact_loss_fn(pred_cont, true_cont, seq_mask)

    l_rec  = pred_coords.new_zeros(1)
    for i, ci in enumerate(all_coords[:-1]):
        w  = cfg.W_RECYCLE * (i + 1) / len(all_coords)
        mi = coord_mask.bool().unsqueeze(-1).expand_as(ci)
        if mi.any():
            l_rec = l_rec + w * F.mse_loss(ci[mi], true_coords[mi])

    total = (cfg.W_COORD   * (l_mse + l_fape) +
             cfg.W_TM      * l_tm +
             cfg.W_DIST    * l_dist +
             cfg.W_CONTACT * l_cont +
             l_rec)

    parts = dict(
        mse=l_mse.item(), tm=l_tm.item(), fape=l_fape.item(),
        dist=l_dist.item(), contact=l_cont.item(), recycle=l_rec.item(),
    )
    return total, parts


# ─────────────────────────────────────────────────────────────
# SCHEDULER  (unchanged from v2)
# ─────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.opt      = optimizer
        self.warmup   = warmup_steps
        self.total    = total_steps
        self.min_lr   = min_lr
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self._step    = 0

    def step(self):
        self._step += 1
        s = self._step
        for i, g in enumerate(self.opt.param_groups):
            base = self.base_lrs[i]
            if s < self.warmup:
                lr = base * s / max(self.warmup, 1)
            else:
                p  = (s - self.warmup) / max(self.total - self.warmup, 1)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (1 + math.cos(math.pi * p))
            g['lr'] = lr

    def get_last_lr(self):
        return [g['lr'] for g in self.opt.param_groups]


# ─────────────────────────────────────────────────────────────
# TRAIN / EVAL LOOPS  (unchanged from v2)
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler, epoch, device):
    model.train()
    total, n = 0.0, 0
    agg = {k: 0.0 for k in ['mse','tm','fape','dist','contact','recycle']}
    n_batches = len(loader)

    # Print a metrics header once per epoch so Kaggle's truncated tqdm still
    # shows progress even if the postfix gets cut off.
    print(f"  {'step':>6}  {'loss':>8}  {'mse':>7}  {'tm':>7}  "
          f"{'fape':>7}  {'gnorm':>7}  {'lr':>9}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*9}")

    bar = tqdm(
        loader,
        desc=f"  Epoch {epoch:02d} [train]",
        leave=True,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    )
    # Print every ~10% of steps (at least every 10 steps, at most every 50)
    log_every = max(1, min(50, n_batches // 10))

    for step, batch in enumerate(bar, 1):
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.MIXED_PREC):
            outputs = model(batch, device=device)
            loss, parts = multi_task_loss(outputs, batch, device)

        if cfg.MIXED_PREC:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

        scheduler.step()

        total += loss.item()
        n     += 1
        for k in agg:
            agg[k] += parts[k]

        lr_now   = scheduler.get_last_lr()[0]
        avg_loss = total / n
        avg_mse  = agg['mse'] / n
        avg_tm   = agg['tm'] / n
        avg_fape = agg['fape'] / n

        bar.set_postfix(
            loss  = f"{avg_loss:.4f}",
            mse   = f"{avg_mse:.3f}",
            tm    = f"{avg_tm:.3f}",
            fape  = f"{avg_fape:.3f}",
            gnorm = f"{float(grad_norm):.2f}",
            lr    = f"{lr_now:.1e}",
            step  = f"{step}/{n_batches}",
        )

        # Plain print fallback — always visible in Kaggle notebook cell output
        if step % log_every == 0 or step == n_batches:
            print(f"  {step:>6}/{n_batches:<6}  {avg_loss:>8.4f}  "
                  f"{avg_mse:>7.4f}  {avg_tm:>7.4f}  "
                  f"{avg_fape:>7.4f}  {float(grad_norm):>7.3f}  {lr_now:>9.2e}")

    return total / n, {k: v / n for k, v in agg.items()}


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    bar = tqdm(loader, desc="  Validation", leave=True,
               dynamic_ncols=True,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]{postfix}")
    for batch in bar:
        with autocast(enabled=cfg.MIXED_PREC):
            outputs = model(batch, device=device)
            loss, _ = multi_task_loss(outputs, batch, device)
        total += loss.item()
        n     += 1
        bar.set_postfix(val_loss=f"{total/n:.4f}")
    val_loss = total / n
    print(f"  Val loss: {val_loss:.5f}")
    return val_loss


# ─────────────────────────────────────────────────────────────
# TRAINING ENTRY POINT
# ─────────────────────────────────────────────────────────────
def run_training(
    train_frac : float = cfg.TRAIN_FRAC,
    epochs     : int   = cfg.EPOCHS,
    batch_size : int   = cfg.BATCH_SIZE,
):
    """
    Main training function — v3 with feature pre-caching.

    WHAT HAPPENS ON FIRST RUN:
      1. Load all CIF files and compute features (MSA, secondary structure,
         geometry, RBF) ONCE using GPU where possible.
      2. Save to /kaggle/working/feat_cache/{split}/{tid}.npz
      3. Train using CachedRNADataset (pure file I/O per step).

    WHAT HAPPENS ON SUBSEQUENT RUNS:
      1. Cache check — all .npz files present → skip step 1 entirely.
      2. Jump straight to training.
    """
    print(f"\n{'='*70}")
    print(f"  RNA 3D FOLDING v3 — SE(3) + Feature Pre-Cache")
    print(f"  train_frac={train_frac}  epochs={epochs}  batch={batch_size}")
    print(f"{'='*70}")

    # ── Model + device ──
    model, device = build_model_dual_gpu()
    n_gpus   = torch.cuda.device_count()
    eff_batch = batch_size * max(n_gpus, 1)
    # Note: build_model_dual_gpu() already prints the GPU info line
    print(f"  Effective batch size: {eff_batch}  ({batch_size} × {max(n_gpus,1)} GPU)")

    # ── Load CSVs ──
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    valid_df = pd.read_csv(cfg.VALID_CSV)

    if train_frac < 1.0:
        n_use    = max(1, int(len(train_df) * train_frac))
        train_df = train_df.sample(n=n_use, random_state=cfg.SEED).reset_index(drop=True)
        print(f"  Training on {n_use}/{len(pd.read_csv(cfg.TRAIN_CSV))} samples ({train_frac*100:.0f}%)")
    else:
        print(f"  Training on all {len(train_df)} samples")
    print(f"  Validation: {len(valid_df)} samples")

    # ─────────────────────────────────────────────────────────
    # PRE-COMPUTE FEATURES (runs once; skipped if cache exists)
    # ─────────────────────────────────────────────────────────
    print(f"\n  ── Phase 1: Feature Pre-Computation ───────────────────")
    print(f"  Cache dir: {cfg.CACHE_DIR}")

    train_rows = _build_row_dicts(train_df, split='train')
    valid_rows = _build_row_dicts(valid_df, split='val')

    precompute_split(train_rows, cfg.CACHE_DIR, cfg.MAX_LEN, cfg.MSA_DIR,
                     device=device, desc="Train")
    precompute_split(valid_rows, cfg.CACHE_DIR, cfg.MAX_LEN, cfg.MSA_DIR,
                     device=device, desc="Val")

    # ─────────────────────────────────────────────────────────
    # BUILD DATASETS FROM CACHE
    # ─────────────────────────────────────────────────────────
    print(f"\n  ── Phase 2: Training ───────────────────────────────────")
    train_ids = train_df['target_id'].tolist()
    valid_ids = valid_df['target_id'].tolist()

    train_ds = CachedRNADataset(cfg.CACHE_DIR, 'train', train_ids)
    valid_ds = CachedRNADataset(cfg.CACHE_DIR, 'val',   valid_ids)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = cfg.PIN_MEMORY,
        drop_last   = True,
        persistent_workers = (cfg.NUM_WORKERS > 0),
        prefetch_factor    = 2 if cfg.NUM_WORKERS > 0 else None,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = cfg.PIN_MEMORY,
        persistent_workers = (cfg.NUM_WORKERS > 0),
        prefetch_factor    = 2 if cfg.NUM_WORKERS > 0 else None,
    )

    # ── Optimizer ──
    base_model = model.module if hasattr(model, 'module') else model
    pair_params  = [p for n, p in base_model.named_parameters() if 'pair_embed' in n]
    other_params = [p for n, p in base_model.named_parameters() if 'pair_embed' not in n]
    optimizer = AdamW([
        {'params': pair_params,  'lr': cfg.LR_PAIR},
        {'params': other_params, 'lr': cfg.LR},
    ], weight_decay=cfg.WEIGHT_DECAY)

    total_steps = epochs * len(train_loader)
    scheduler   = WarmupCosineScheduler(optimizer, cfg.WARMUP_STEPS, total_steps)
    scaler      = GradScaler(enabled=cfg.MIXED_PREC)

    best_val  = float('inf')
    history   = []
    ckpt_path = os.path.join(cfg.OUT_DIR, 'best_rna_se3_v3.pt')

    print(f"\n  {'─'*70}")
    print(f"  Starting training for {epochs} epochs  |  {len(train_loader)} steps/epoch")
    print(f"  {'─'*70}\n")

    for epoch in range(1, epochs + 1):
        print(f"\n{'━'*70}")
        print(f"  EPOCH {epoch}/{epochs}")
        print(f"{'━'*70}")

        train_loss, parts = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, epoch, device)
        val_loss = eval_epoch(model, valid_loader, device)

        lrs = scheduler.get_last_lr()
        print(
            f"\n  ┌─ Epoch {epoch:3d}/{epochs} Summary {'─'*38}\n"
            f"  │  Train loss : {train_loss:.5f}   Val loss : {val_loss:.5f}\n"
            f"  │  ├ mse={parts['mse']:.4f}  tm={parts['tm']:.4f}  fape={parts['fape']:.4f}\n"
            f"  │  └ dist={parts['dist']:.4f}  contact={parts['contact']:.4f}  recycle={parts['recycle']:.4f}\n"
            f"  │  LR (pair/other): {lrs[0]:.2e} / {lrs[-1]:.2e}\n"
            f"  └{'─'*56}"
        )

        history.append({'epoch': epoch, 'train': train_loss, 'val': val_loss, **parts})

        if val_loss < best_val:
            best_val = val_loss
            state = base_model.state_dict()
            torch.save({'epoch': epoch, 'model': state,
                        'val_loss': val_loss, 'history': history}, ckpt_path)
            print(f"     ✅  Saved best model (val={best_val:.4f})")

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    pd.DataFrame(history).to_csv(
        os.path.join(cfg.OUT_DIR, 'training_history_v3.csv'), index=False)
    print(f"\n  Training complete. Best val loss: {best_val:.4f}")
    return model, history, device


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_dataset(model, csv_path: str, is_test: bool,
                    device, batch_size=cfg.BATCH_SIZE) -> pd.DataFrame:
    """
    Predict for a CSV file.  Features are pre-cached (test split if needed).
    """
    model.eval()
    df    = pd.read_csv(csv_path)
    split = 'test' if is_test else 'val'

    # Cache test features if not done yet
    rows = _build_row_dicts(df, split=split)
    precompute_split(rows, cfg.CACHE_DIR, cfg.MAX_LEN, cfg.MSA_DIR,
                     device=device, desc=split.capitalize())

    ids    = df['target_id'].tolist()
    ds     = CachedRNADataset(cfg.CACHE_DIR, split, ids)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=(cfg.NUM_WORKERS > 0),
        prefetch_factor=2 if cfg.NUM_WORKERS > 0 else None,
    )
    rows_out = []
    for batch in tqdm(loader, desc=f"  Predicting {Path(csv_path).stem}"):
        with autocast(enabled=cfg.MIXED_PREC):
            outputs = model(batch, device=device)
        coords = outputs['coords'].cpu().float().numpy()
        for b, tid in enumerate(batch['target_id']):
            L = batch['seq_len'][b]
            for i in range(L):
                rows_out.append({
                    'target_id': tid, 'resid': i + 1,
                    'x_1': float(coords[b, i, 0]),
                    'y_1': float(coords[b, i, 1]),
                    'z_1': float(coords[b, i, 2]),
                })
    return pd.DataFrame(rows_out)


def run_inference(model=None, device=None):
    from rna_model_se3_v2 import RNAFoldSE3

    if model is None:
        # Try v3 checkpoint first, fall back to v2
        for ckpt_name in ['best_rna_se3_v3.pt', 'best_rna_se3_v2.pt']:
            ckpt = os.path.join(cfg.OUT_DIR, ckpt_name)
            if Path(ckpt).exists():
                break
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"No checkpoint found in {cfg.OUT_DIR}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base   = RNAFoldSE3().to(device)
        state  = torch.load(ckpt, map_location=device)
        base.load_state_dict(state['model'])
        if torch.cuda.device_count() >= 2:
            model = nn.DataParallel(base, device_ids=list(range(torch.cuda.device_count())))
        else:
            model = base
        model = model.to(device)
        print(f"  Loaded checkpoint ({ckpt_name}) → epoch={state['epoch']}  val={state['val_loss']:.4f}")

    if device is None:
        device = next(model.parameters()).device

    for split, cpath, is_t in [
        ('validation', cfg.VALID_CSV, False),
        ('test',       cfg.TEST_CSV,  True),
    ]:
        if not Path(cpath).exists():
            print(f"  [skip] {cpath} not found")
            continue
        df  = predict_dataset(model, cpath, is_t, device)
        out = os.path.join(cfg.OUT_DIR, f'predictions_{split}.csv')
        df.to_csv(out, index=False)
        print(f"  Saved {split} predictions → {out}  ({len(df):,} rows)")


# ─────────────────────────────────────────────────────────────
# EVALUATION  (unchanged from v2)
# ─────────────────────────────────────────────────────────────
def evaluate_on_validation():
    pred_path = os.path.join(cfg.OUT_DIR, 'predictions_validation.csv')
    if not Path(pred_path).exists():
        print("No validation predictions found."); return

    pred_df  = pd.read_csv(pred_path)
    valid_df = pd.read_csv(cfg.VALID_CSV)
    results  = []

    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Evaluating"):
        tid    = row['target_id']
        seq    = str(row['sequence'])
        result = load_cif_coords(tid)
        if result is None: continue

        cif_seq, cif_coords = result
        aligned = align_coords(seq, cif_seq, cif_coords, cfg.MAX_LEN)
        valid   = ~np.isnan(aligned[:, 0])
        if valid.sum() < 3: continue
        true_c  = aligned[valid]

        pg = pred_df[pred_df['target_id'] == tid].sort_values('resid')
        if len(pg) == 0: continue
        pred_c = pg[['x_1','y_1','z_1']].values.astype(np.float32)
        mn     = min(len(true_c), len(pred_c))
        true_c, pred_c = true_c[:mn], pred_c[:mn]

        def kabsch(P, Q):
            P = P - P.mean(0); Q = Q - Q.mean(0)
            H = P.T @ Q
            U, S, Vt = np.linalg.svd(H)
            D = np.diag([1, 1, np.linalg.det(Vt.T @ U.T)])
            return P @ (Vt.T @ D @ U.T).T, Q

        P_rot, Q = kabsch(pred_c, true_c)
        rmsd = float(np.sqrt(((P_rot - Q)**2).sum(-1).mean()))
        d0   = max(1.24*(mn-15)**(1/3)-1.8, 0.5) if mn > 21 else 0.5
        tm   = float((1/(1+((P_rot-Q)**2).sum(-1)/d0**2)).mean())
        results.append({'target_id': tid, 'L': mn, 'RMSD': rmsd, 'TM': tm})

    df = pd.DataFrame(results)
    if len(df):
        print(f"\n  {'='*50}")
        print(f"  Validation ({len(df)} targets)")
        print(f"  Mean RMSD : {df['RMSD'].mean():.3f} Å")
        print(f"  Mean TM   : {df['TM'].mean():.4f}")
        print(f"  TM > 0.5  : {(df['TM'] > 0.5).mean()*100:.1f}%")
        print(f"  {'='*50}")
        df.to_csv(os.path.join(cfg.OUT_DIR, 'validation_metrics_v3.csv'), index=False)
    return df


# ─────────────────────────────────────────────────────────────
# SUBMISSION
# ─────────────────────────────────────────────────────────────
def refine_and_submit(pred_path: str, out_path: str):
    try:
        from rna_physics_refinement import post_process_predictions, format_submission
        df  = pd.read_csv(pred_path)
        df  = post_process_predictions(df, apply_physics=True)
        sub = format_submission(df, out_path)
    except ImportError:
        df  = pd.read_csv(pred_path)
        df['ID'] = df['target_id'] + '_' + df['resid'].astype(str)
        sub = df[['ID', 'x_1', 'y_1', 'z_1']]
        sub.to_csv(out_path, index=False)
        print(f"  Submission saved → {out_path}")
    return sub


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    mode       = sys.argv[1] if len(sys.argv) > 1 else "full"
    train_frac = float(sys.argv[2]) if len(sys.argv) > 2 else cfg.TRAIN_FRAC
    epochs     = int(sys.argv[3])   if len(sys.argv) > 3 else cfg.EPOCHS
    batch_size = int(sys.argv[4])   if len(sys.argv) > 4 else cfg.BATCH_SIZE

    model, device = None, None

    if mode in ("train", "full"):
        model, history, device = run_training(
            train_frac=train_frac, epochs=epochs, batch_size=batch_size)

    if mode in ("infer", "full"):
        run_inference(model, device)

    if mode in ("eval", "full"):
        evaluate_on_validation()

    if mode in ("submit", "full"):
        test_pred = os.path.join(cfg.OUT_DIR, 'predictions_test.csv')
        sub_path  = os.path.join(cfg.OUT_DIR, 'submission.csv')
        if Path(test_pred).exists():
            refine_and_submit(test_pred, sub_path)

    print("\n🎉 Done!")
