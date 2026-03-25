"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RNA 3D FOLDING — TRAINING PIPELINE v2                                       ║
║                                                                              ║
║  Key upgrades over v1:                                                       ║
║   • Dual GPU (DataParallel) support                                          ║
║   • Configurable % of training data                                          ║
║   • Secondary structure features fully integrated                            ║
║   • Case-insensitive PDB / MSA file lookup                                  ║
║   • Long-sequence support via chunked dataset mode                           ║
║   • Improved collate for variable-length sequences                           ║
║   • Per-epoch TM-score logging                                               ║
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
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from Bio.PDB import MMCIFParser

# ── local modules ──
from rna_features_v2 import (
    build_all_features, find_msa_file,
    chunk_sequence, stitch_coords,
)
from rna_model_se3_v2 import RNAFoldSE3, build_model_dual_gpu, cfg as ModelCfg

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# TRAINING CONFIG  ← adjust these to control the run
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

    # ── DATA ─────────────────────────────────────────────────
    TRAIN_FRAC   = 1.0      # use 100% of training data (set 0.5 for 50%, etc.)
    MAX_LEN      = ModelCfg.MAX_LEN   # 512 — pad/crop length per chunk

    # ── MODEL ────────────────────────────────────────────────
    BATCH_SIZE   = 4         # per GPU; with 2 GPUs effective = 8
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

    # ── MISC ─────────────────────────────────────────────────
    SEED         = 42
    NUM_WORKERS  = 2

cfg = TrainConfig()

# ─────────────────────────────────────────────────────────────
# CONSTANTS
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
# CIF COORDINATE LOADER  (case-insensitive)
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


def coords_to_dist_bins(coords: np.ndarray, L: int) -> np.ndarray:
    Lr   = min(len(coords), L)
    dist = np.zeros((L, L), np.float32)
    valid = ~np.isnan(coords[:Lr, 0])
    vi   = np.where(valid)[0]
    for ii, i in enumerate(vi):
        for j in vi[ii+1:]:
            d = np.linalg.norm(coords[i] - coords[j])
            dist[i, j] = dist[j, i] = d
    bins = np.digitize(dist, BIN_EDGES) - 1
    return np.clip(bins, 0, N_BINS - 1).astype(np.int64)


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class RNADataset(Dataset):
    """
    Returns per-sample dict with ALL features.
    At inference (is_test=True): geometric features are zeroed.
    Secondary structure is always computed from sequence.
    """

    def __init__(self, csv_path: str, is_test: bool = False):
        self.df      = pd.read_csv(csv_path)
        self.is_test = is_test
        self.L       = cfg.MAX_LEN

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        tid     = row['target_id']
        seq     = str(row['sequence'])
        L_seq   = min(len(seq), self.L)
        L       = self.L

        # ── Token IDs ──
        seq_ids = torch.tensor(
            [VOCAB.get(c, VOCAB['<UNK>']) for c in seq[:L]] +
            [VOCAB['<PAD>']] * (L - L_seq),
            dtype=torch.long,
        )
        seq_mask = torch.zeros(L, dtype=torch.bool)
        seq_mask[:L_seq] = True

        # ── Load coords ──
        coords_gt   = np.full((L, 3), np.nan, np.float32)
        coord_mask  = np.zeros(L, np.float32)
        dist_bins_t = np.zeros((L, L), np.int64)
        contact_t   = np.zeros((L, L), np.float32)

        if not self.is_test:
            result = load_cif_coords(tid)
            if result is not None:
                cif_seq, cif_coords = result
                aligned = align_coords(seq, cif_seq, cif_coords, L)
                valid   = ~np.isnan(aligned[:, 0])
                coords_gt[valid]  = aligned[valid]
                coord_mask[valid] = 1.0
                dist_bins_t       = coords_to_dist_bins(aligned, L)
                # contact target (< 8 Å)
                dist_full = np.zeros((L, L), np.float32)
                vi = np.where(valid)[0]
                for ii, i2 in enumerate(vi):
                    for j2 in vi[ii+1:]:
                        d = np.linalg.norm(coords_gt[i2] - coords_gt[j2])
                        dist_full[i2, j2] = dist_full[j2, i2] = d
                contact_t = (dist_full < 8.0).astype(np.float32)
                np.fill_diagonal(contact_t, 0)

        # ── Feature extraction ──
        # coords=None at inference → geo features will be zeros
        coords_for_feat = (coords_gt if coord_mask.sum() > 0 and not self.is_test
                           else None)
        feats = build_all_features(
            seq        = seq,
            target_id  = tid,
            coords     = coords_for_feat,
            msa_dir    = cfg.MSA_DIR,
            max_len    = L,
            is_inference = self.is_test,
        )

        return {
            # ── inputs ──
            'seq_ids'    : seq_ids,
            'seq_mask'   : seq_mask,
            'f1'         : torch.from_numpy(feats['f1'][:L]),
            'dihed'      : torch.from_numpy(feats['dihed']),
            'ss_pair'    : torch.from_numpy(feats['ss_pair']),
            'dist_rbf'   : torch.from_numpy(feats['dist_rbf']),
            'dist_bins'  : torch.from_numpy(feats['dist_bins']),
            'orient'     : torch.from_numpy(feats['orient']),
            'rel_pos'    : torch.from_numpy(feats['rel_pos']),
            'MIp'        : torch.from_numpy(feats['MIp']),
            'FNp'        : torch.from_numpy(feats['FNp']),
            'contact_ss' : torch.from_numpy(feats['contact_ss']),
            'pair_type'  : torch.from_numpy(feats['pair_type']),
            # ── targets ──
            'coords'      : torch.from_numpy(coords_gt),
            'coord_mask'  : torch.from_numpy(coord_mask),
            'dist_bins_t' : torch.from_numpy(dist_bins_t),
            'contact_t'   : torch.from_numpy(contact_t),
            # ── metadata ──
            'seq_len'    : L_seq,
            'target_id'  : tid,
        }


def collate_fn(batch):
    scalar_keys = {'seq_len', 'target_id'}
    tensor_keys = [k for k in batch[0] if k not in scalar_keys]
    out = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    out['seq_len']   = [b['seq_len']   for b in batch]
    out['target_id'] = [b['target_id'] for b in batch]
    return out


# ─────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
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
    pm  = mask.unsqueeze(2).float() * mask.unsqueeze(1).float()
    true = true.to(pred.device)
    pw  = torch.tensor(5.0, device=pred.device)
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

    total = (cfg.W_COORD  * (l_mse + l_fape) +
             cfg.W_TM     * l_tm +
             cfg.W_DIST   * l_dist +
             cfg.W_CONTACT * l_cont +
             l_rec)

    parts = dict(
        mse=l_mse.item(), tm=l_tm.item(), fape=l_fape.item(),
        dist=l_dist.item(), contact=l_cont.item(), recycle=l_rec.item(),
    )
    return total, parts


# ─────────────────────────────────────────────────────────────
# SCHEDULER
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
# TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler, epoch, device):
    model.train()
    total, n = 0.0, 0
    agg = {k: 0.0 for k in ['mse','tm','fape','dist','contact','recycle']}

    bar = tqdm(loader, desc=f"Epoch {epoch:02d}", leave=False)
    for batch in bar:
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.MIXED_PREC):
            outputs = model(batch, device=device)
            loss, parts = multi_task_loss(outputs, batch, device)

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

        total += loss.item()
        n     += 1
        for k in agg:
            agg[k] += parts[k]
        bar.set_postfix(loss=f"{total/n:.4f}", tm=f"{agg['tm']/n:.3f}")

    return total / n, {k: v / n for k, v in agg.items()}


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="  Val", leave=False):
        with autocast(enabled=cfg.MIXED_PREC):
            outputs = model(batch, device=device)
            loss, _ = multi_task_loss(outputs, batch, device)
        total += loss.item()
        n     += 1
    return total / n


# ─────────────────────────────────────────────────────────────
# TRAINING ENTRY POINT
# ─────────────────────────────────────────────────────────────
def run_training(
    train_frac : float = cfg.TRAIN_FRAC,
    epochs     : int   = cfg.EPOCHS,
    batch_size : int   = cfg.BATCH_SIZE,
):
    """
    Main training function.

    Parameters
    ----------
    train_frac : float, 0 < x ≤ 1.0
        Fraction of training data to use. E.g. 0.5 uses 50%.
    epochs     : int
        Number of training epochs.
    batch_size : int
        Batch size per GPU.
    """
    print(f"\n{'='*70}")
    print(f"  RNA 3D FOLDING v2 — SE(3) Equivariant Pipeline")
    print(f"  train_frac={train_frac}  epochs={epochs}  batch={batch_size}")
    print(f"{'='*70}")

    # ── Model + device ──
    model, device = build_model_dual_gpu()
    n_gpus = torch.cuda.device_count()
    eff_batch = batch_size * max(n_gpus, 1)
    print(f"  Effective batch size: {eff_batch}  ({batch_size} × {max(n_gpus,1)} GPU)")

    # ── Datasets ──
    full_train_ds = RNADataset(cfg.TRAIN_CSV, is_test=False)
    valid_ds      = RNADataset(cfg.VALID_CSV, is_test=False)

    if train_frac < 1.0:
        n_use  = max(1, int(len(full_train_ds) * train_frac))
        idx    = list(range(len(full_train_ds)))
        random.shuffle(idx)
        train_ds = Subset(full_train_ds, idx[:n_use])
        print(f"  Training on {n_use}/{len(full_train_ds)} samples ({train_frac*100:.0f}%)")
    else:
        train_ds = full_train_ds
        print(f"  Training on all {len(train_ds)} samples")

    print(f"  Validation: {len(valid_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS,
        pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS,
    )

    # ── Optimizer ──
    # Unwrap DataParallel for named_parameters
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
    ckpt_path = os.path.join(cfg.OUT_DIR, 'best_rna_se3_v2.pt')

    print(f"\n  {'─'*60}")
    print(f"  Starting training for {epochs} epochs")
    print(f"  {'─'*60}")

    for epoch in range(1, epochs + 1):
        train_loss, parts = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, epoch, device)
        val_loss = eval_epoch(model, valid_loader, device)

        lrs = scheduler.get_last_lr()
        print(
            f"  Ep {epoch:3d}/{epochs} | "
            f"train={train_loss:.4f} "
            f"(mse={parts['mse']:.3f} tm={parts['tm']:.3f} "
            f"fape={parts['fape']:.3f} dist={parts['dist']:.3f}) | "
            f"val={val_loss:.4f} | lr={lrs[0]:.2e}"
        )

        history.append({'epoch': epoch, 'train': train_loss, 'val': val_loss, **parts})

        if val_loss < best_val:
            best_val = val_loss
            # Save underlying model weights (not DataParallel wrapper)
            state = base_model.state_dict()
            torch.save({'epoch': epoch, 'model': state,
                        'val_loss': val_loss, 'history': history}, ckpt_path)
            print(f"     ✅  Saved best model (val={best_val:.4f})")

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    pd.DataFrame(history).to_csv(
        os.path.join(cfg.OUT_DIR, 'training_history_v2.csv'), index=False)
    print(f"\n  Training complete. Best val loss: {best_val:.4f}")
    return model, history, device


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_dataset(model, csv_path: str, is_test: bool,
                    device, batch_size=cfg.BATCH_SIZE) -> pd.DataFrame:
    model.eval()
    ds     = RNADataset(csv_path, is_test=is_test)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS,
    )
    rows = []
    for batch in tqdm(loader, desc=f"  Predicting {Path(csv_path).stem}"):
        with autocast(enabled=cfg.MIXED_PREC):
            outputs = model(batch, device=device)
        coords = outputs['coords'].cpu().float().numpy()   # (B, L, 3)
        for b, tid in enumerate(batch['target_id']):
            L = batch['seq_len'][b]
            for i in range(L):
                rows.append({
                    'target_id': tid, 'resid': i + 1,
                    'x_1': float(coords[b, i, 0]),
                    'y_1': float(coords[b, i, 1]),
                    'z_1': float(coords[b, i, 2]),
                })
    return pd.DataFrame(rows)


def run_inference(model=None, device=None):
    """
    Load the best checkpoint and run inference on validation + test sets.
    At inference time, geometric features are ZERO; only sequence, MSA,
    and predicted secondary structure are used.
    """
    from rna_model_se3_v2 import RNAFoldSE3

    if model is None:
        ckpt = os.path.join(cfg.OUT_DIR, 'best_rna_se3_v2.pt')
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"No checkpoint found at {ckpt}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base   = RNAFoldSE3().to(device)
        state  = torch.load(ckpt, map_location=device)
        base.load_state_dict(state['model'])
        # re-wrap if multi-GPU
        if torch.cuda.device_count() >= 2:
            model = nn.DataParallel(base, device_ids=list(range(torch.cuda.device_count())))
        else:
            model = base
        model = model.to(device)
        print(f"  Loaded checkpoint → epoch={state['epoch']}  val={state['val_loss']:.4f}")

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
# EVALUATION
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

        # Kabsch
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
        df.to_csv(os.path.join(cfg.OUT_DIR, 'validation_metrics_v2.csv'), index=False)
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

    # ── Example: customise from command line ──
    # python rna_train_v2.py train 0.5 20 4
    #   → 50% data, 20 epochs, batch 4

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
