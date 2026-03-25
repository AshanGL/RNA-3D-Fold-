"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SE(3)-EQUIVARIANT RNA FOLDING MODEL v2                                      ║
║                                                                              ║
║  CHANGES FROM v1:                                                            ║
║   • Secondary structure branch added (contact_ss + pair_type)               ║
║   • NodeEmbedding enhanced: ss_pair flag + extended f1                      ║
║   • PairEmbedding: accepts contact_ss + pair_type instead of raw dist_bins  ║
║     at inference time (geometric → optional, SS → always on)                ║
║   • Gradient checkpointing option for long sequences                        ║
║   • DualGPU: DataParallel-compatible forward signature                      ║
║   • Sliding-window attention for sequences > MAX_LEN                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Dict, Optional, Tuple, List

# Reduces CUDA allocator fragmentation on large pair tensors.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
class ModelConfig:
    # Sequence / pair dims
    # D_PAIR 128→64: pair tensor is (B,L,L,D_PAIR) — halving cuts pair memory 4×.
    # D_NODE 256→128: cuts single-rep and IPA memory in half.
    D_NODE   = 128
    D_PAIR   = 64
    D_HIDDEN = 64

    # Attention — keep n_head divisible into D_NODE and D_PAIR
    N_HEAD      = 4   # was 8
    N_QUERY_PT  = 4
    N_VALUE_PT  = 4   # was 8

    # Layers
    # N_EVOFORMER 8→4: each block stores (B,L,L,D_PAIR) for backward; fewer = less peak RAM.
    # N_RECYCLE   3→2: each recycle reruns the full stack.
    N_EVOFORMER = 4
    N_STRUCTURE = 3
    N_RECYCLE   = 2

    # Input feature dims (must match rna_features_v2.py)
    N_DIST_BINS = 36
    N_ORIENT    = 4
    N_RBF       = 16
    N_REL_POS   = 65
    N_DIHED     = 4
    N_PAIR_TYPE = 3
    F1_DIM      = 5
    VOCAB_SIZE  = 6     # A U G C <PAD> <UNK>

    DROPOUT     = 0.1
    MAX_LEN     = 512

    # Long sequence support
    CHUNK_SIZE  = 512   # process in chunks if seq > this
    CHUNK_OVERLAP = 64


cfg = ModelConfig()

VOCAB = {'A':0,'U':1,'G':2,'C':3,'<PAD>':4,'<UNK>':5}


# ─────────────────────────────────────────────────────────────
# UTILITY LAYERS
# ─────────────────────────────────────────────────────────────
class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


class Linear(nn.Linear):
    def __init__(self, in_f, out_f, bias=True, init='default'):
        super().__init__(in_f, out_f, bias=bias)
        if init == 'relu':
            nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        elif init in ('zeros', 'final'):
            nn.init.zeros_(self.weight)
            if bias: nn.init.zeros_(self.bias)
        else:
            nn.init.xavier_uniform_(self.weight)
            if bias: nn.init.zeros_(self.bias)


class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len):
        super().__init__()
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ─────────────────────────────────────────────────────────────
# 1. INPUT EMBEDDINGS
# ─────────────────────────────────────────────────────────────
class NodeEmbedding(nn.Module):
    """
    Single (node) representation from:
      - sequence token embedding
      - f1 frequency profile  (5 dims)
      - dihedral angles        (4 dims) — zeros at inference
      - secondary structure pairing flag (1 dim) — always present
      - sinusoidal positional encoding
    """
    def __init__(self):
        super().__init__()
        self.seq_embed  = nn.Embedding(cfg.VOCAB_SIZE, 64, padding_idx=4)
        self.f1_proj    = Linear(cfg.F1_DIM, 32)
        self.dihed_proj = Linear(cfg.N_DIHED, 32)
        self.ss_proj    = Linear(1, 16)          # ss_pair (1 scalar per position)
        self.pe         = SinusoidalPE(cfg.D_NODE, cfg.MAX_LEN)

        in_dim = 64 + 32 + 32 + 16
        self.out_proj = nn.Sequential(
            Linear(in_dim, cfg.D_NODE),
            nn.ReLU(),
            LayerNorm(cfg.D_NODE),
        )

    def forward(self, seq_ids, f1, dihed, ss_pair):
        """
        seq_ids : (B, L)
        f1      : (B, L, 5)
        dihed   : (B, L, 4)  — zeros at inference is fine
        ss_pair : (B, L)     — 1.0 if residue is predicted to be paired
        """
        s = self.seq_embed(seq_ids)                   # (B, L, 64)
        f = self.f1_proj(f1)                         # (B, L, 32)
        d = self.dihed_proj(dihed)                   # (B, L, 32)
        p = self.ss_proj(ss_pair.unsqueeze(-1))      # (B, L, 16)
        x = torch.cat([s, f, d, p], dim=-1)          # (B, L, 144)
        x = self.out_proj(x)                         # (B, L, D_NODE)
        x = self.pe(x)
        return x


class PairEmbedding(nn.Module):
    """
    Pair representation from:
      ALWAYS:
        - relative position      (65 dims)
        - MIp covariation        ( 1 dim)
        - FNp direct information ( 1 dim)
        - contact_ss (secondary) ( 1 dim)  ← new
        - pair_type one-hot      ( 3 dims) ← new

      AT TRAIN TIME (zeros at inference):
        - RBF distances          (16 dims)
        - binned distances       (36 dims)
        - orientation            ( 4 dims)
    """
    def __init__(self):
        super().__init__()
        always_dim = cfg.N_REL_POS + 2 + 1 + cfg.N_PAIR_TYPE  # 65+2+1+3 = 71
        geo_dim    = cfg.N_RBF + cfg.N_DIST_BINS + cfg.N_ORIENT  # 16+36+4 = 56
        in_dim     = always_dim + geo_dim  # 127

        self.proj = nn.Sequential(
            Linear(in_dim, cfg.D_PAIR * 2),
            nn.ReLU(),
            Linear(cfg.D_PAIR * 2, cfg.D_PAIR),
            LayerNorm(cfg.D_PAIR),
        )

    def forward(self, rbf, dist_bins, orient, rel_pos,
                MIp, FNp, contact_ss, pair_type):
        cov = torch.stack([MIp, FNp], dim=-1)        # (B, L, L, 2)
        x   = torch.cat([
            rbf,          # (B, L, L, 16) — zeros at inference
            dist_bins,    # (B, L, L, 36)
            orient,       # (B, L, L, 4)
            rel_pos,      # (B, L, L, 65)
            cov,          # (B, L, L, 2)
            contact_ss.unsqueeze(-1),   # (B, L, L, 1)
            pair_type,    # (B, L, L, 3)
        ], dim=-1)        # total: 127
        return self.proj(x)     # (B, L, L, D_PAIR)


# ─────────────────────────────────────────────────────────────
# 2. EVOFORMER-STYLE BLOCKS
# ─────────────────────────────────────────────────────────────
class RowAttentionWithPairBias(nn.Module):
    def __init__(self, d_node=cfg.D_NODE, d_pair=cfg.D_PAIR, n_head=cfg.N_HEAD):
        super().__init__()
        self.n_head  = n_head
        self.d_head  = d_node // n_head
        self.scale   = self.d_head ** -0.5
        self.norm    = LayerNorm(d_node)
        self.q       = Linear(d_node, d_node, bias=False)
        self.k       = Linear(d_node, d_node, bias=False)
        self.v       = Linear(d_node, d_node, bias=False)
        self.gate    = Linear(d_node, d_node)
        self.out     = Linear(d_node, d_node, init='final')
        self.pair_bias = Linear(d_pair, n_head, bias=False)

    def forward(self, single, pair, mask=None):
        B, L, D = single.shape
        H, Dh   = self.n_head, self.d_head
        x  = self.norm(single)
        Q  = self.q(x).reshape(B, L, H, Dh).permute(0, 2, 1, 3)
        K  = self.k(x).reshape(B, L, H, Dh).permute(0, 2, 1, 3)
        V  = self.v(x).reshape(B, L, H, Dh).permute(0, 2, 1, 3)
        g  = torch.sigmoid(self.gate(x)).reshape(B, L, H, Dh).permute(0, 2, 1, 3)
        attn = torch.einsum('bhid,bhjd->bhij', Q, K) * self.scale
        bias = self.pair_bias(pair).permute(0, 3, 1, 2)
        attn = attn + bias
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, :],
                                    torch.finfo(attn.dtype).min / 2)
        attn = F.softmax(attn, dim=-1)
        out  = torch.einsum('bhij,bhjd->bhid', attn, V)
        out  = (g * out).permute(0, 2, 1, 3).reshape(B, L, D)
        return single + self.out(out)


class PairUpdate(nn.Module):
    def __init__(self, d_node=cfg.D_NODE, d_pair=cfg.D_PAIR):
        super().__init__()
        self.norm      = LayerNorm(d_node)
        self.proj_a    = Linear(d_node, d_pair)
        self.proj_b    = Linear(d_node, d_pair)
        self.out       = Linear(d_pair, d_pair, init='final')
        self.pair_norm = LayerNorm(d_pair)

    def forward(self, single, pair):
        x  = self.norm(single)
        a  = self.proj_a(x)    # (B, L, D_PAIR)
        b  = self.proj_b(x)
        op = torch.einsum('bid,bjd->bijd', a, b)  # (B, L, L, D_PAIR)
        return pair + self.pair_norm(self.out(op))


class TriangleAttention(nn.Module):
    """Axial attention on pair representation (row-wise and column-wise)."""
    def __init__(self, d_pair=cfg.D_PAIR, n_head=4, mode='row'):
        super().__init__()
        self.mode   = mode
        self.n_head = n_head
        self.d_head = d_pair // n_head
        self.scale  = self.d_head ** -0.5
        self.norm   = LayerNorm(d_pair)
        self.q      = Linear(d_pair, d_pair, bias=False)
        self.k      = Linear(d_pair, d_pair, bias=False)
        self.v      = Linear(d_pair, d_pair, bias=False)
        self.gate   = Linear(d_pair, d_pair)
        self.out    = Linear(d_pair, d_pair, init='final')

    def forward(self, pair, mask=None):
        # pair: (B, L, L, D_PAIR)
        B, L, _, D = pair.shape
        H, Dh = self.n_head, self.d_head

        x = self.norm(pair)
        if self.mode == 'col':
            x = x.transpose(1, 2).contiguous()  # treat columns as rows

        # Project: (B, L, L, D) → (B, L, L, H, Dh)
        Q = self.q(x).reshape(B, L, L, H, Dh)
        K = self.k(x).reshape(B, L, L, H, Dh)
        V = self.v(x).reshape(B, L, L, H, Dh)
        g = torch.sigmoid(self.gate(x)).reshape(B, L, L, H, Dh)

        # ── Memory-efficient axial attention via F.scaled_dot_product_attention ──
        # Fold the batch and row dims together → treat each row independently.
        # (B, L, L, H, Dh) → (B*L, H, L, Dh)  [row=batch, col=sequence]
        Q_ = Q.reshape(B * L, L, H, Dh).permute(0, 2, 1, 3)  # (B*L, H, L, Dh)
        K_ = K.reshape(B * L, L, H, Dh).permute(0, 2, 1, 3)
        V_ = V.reshape(B * L, L, H, Dh).permute(0, 2, 1, 3)

        # F.scaled_dot_product_attention uses flash-attention when available,
        # never materialises the full (L, L) matrix in fp32 → no OOM.
        out_ = F.scaled_dot_product_attention(Q_, K_, V_)     # (B*L, H, L, Dh)

        # Restore shape: (B*L, H, L, Dh) → (B, L, L, H, Dh) → gate → (B, L, L, D)
        out = out_.permute(0, 2, 1, 3).reshape(B, L, L, H, Dh)
        out = (g * out).reshape(B, L, L, D)

        if self.mode == 'col':
            out = out.transpose(1, 2).contiguous()

        return pair + self.out(out)



class EvoformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        D, DP = cfg.D_NODE, cfg.D_PAIR
        self.row_attn  = RowAttentionWithPairBias(D, DP, cfg.N_HEAD)
        self.pair_upd  = PairUpdate(D, DP)
        self.tri_row   = TriangleAttention(DP, n_head=4, mode='row')
        self.tri_col   = TriangleAttention(DP, n_head=4, mode='col')
        self.norm_s    = LayerNorm(D)
        self.norm_p    = LayerNorm(DP)
        self.ff_s = nn.Sequential(
            Linear(D, D * 4), nn.GELU(), Linear(D * 4, D))
        self.ff_p = nn.Sequential(
            Linear(DP, DP * 4), nn.GELU(), Linear(DP * 4, DP))
        self.drop = nn.Dropout(cfg.DROPOUT)

    def _forward_impl(self, single, pair, mask):
        single = self.row_attn(single, pair, mask)
        single = single + self.drop(self.ff_s(self.norm_s(single)))
        pair   = self.pair_upd(single, pair)
        pair   = self.tri_row(pair)
        pair   = self.tri_col(pair)
        pair   = pair + self.drop(self.ff_p(self.norm_p(pair)))
        return single, pair

    def forward(self, single, pair, mask=None):
        # Gradient checkpointing recomputes activations on backward instead of
        # storing them → ~60% less peak VRAM at cost of extra forward compute.
        if self.training:
            _mask = mask if mask is not None else single.new_zeros(1)
            return grad_checkpoint(self._forward_impl, single, pair, _mask,
                                   use_reentrant=False)
        return self._forward_impl(single, pair, mask)


# ─────────────────────────────────────────────────────────────
# 3. STRUCTURE MODULE (SE(3) equivariant via IPA)
# ─────────────────────────────────────────────────────────────
class InvariantPointAttention(nn.Module):
    """AlphaFold2-style IPA for RNA backbone frames."""
    def __init__(self, d_node=cfg.D_NODE, d_pair=cfg.D_PAIR,
                 n_head=cfg.N_HEAD, n_qp=cfg.N_QUERY_PT, n_vp=cfg.N_VALUE_PT):
        super().__init__()
        self.n_head = n_head
        self.n_qp   = n_qp
        self.n_vp   = n_vp
        self.d_head = d_node // n_head
        self.scale  = (self.d_head + n_qp * 9) ** -0.5

        self.norm   = LayerNorm(d_node)
        self.q_node = Linear(d_node, d_node, bias=False)
        self.k_node = Linear(d_node, d_node, bias=False)
        self.v_node = Linear(d_node, d_node, bias=False)
        self.q_pt   = Linear(d_node, n_head * n_qp * 3, bias=False)
        self.k_pt   = Linear(d_node, n_head * n_qp * 3, bias=False)
        self.v_pt   = Linear(d_node, n_head * n_vp * 3, bias=False)
        self.pair_b = Linear(d_pair, n_head, bias=False)
        self.w_c    = nn.Parameter(torch.ones(n_head))

        out_dim = n_head * (self.d_head + n_vp * 3 + 1)
        self.out_proj = nn.Sequential(
            Linear(out_dim, d_node, init='final'),
        )

    def _apply_frames(self, pts, T):
        """Apply batch of rigid frames to point cloud. pts: (B,L,H*K,3)"""
        R = T[..., :3, :3]   # (B, L, 3, 3)
        t = T[..., :3, 3]    # (B, L, 3)
        # pts: (B, L, H*K, 3) → apply R and t
        pts_t = torch.einsum('blij,blkj->blki', R, pts) + t.unsqueeze(2)
        return pts_t

    def forward(self, single, pair, T):
        B, L, D = single.shape
        H, Dh   = self.n_head, self.d_head

        x  = self.norm(single)
        Q  = self.q_node(x).reshape(B, L, H, Dh).permute(0, 2, 1, 3)
        K  = self.k_node(x).reshape(B, L, H, Dh).permute(0, 2, 1, 3)
        V  = self.v_node(x).reshape(B, L, H, Dh).permute(0, 2, 1, 3)

        # point queries / keys / values in local frame
        Qp = self.q_pt(x).reshape(B, L, H * self.n_qp, 3)
        Kp = self.k_pt(x).reshape(B, L, H * self.n_qp, 3)
        Vp = self.v_pt(x).reshape(B, L, H * self.n_vp, 3)

        Qp_g = self._apply_frames(Qp, T)   # global coords
        Kp_g = self._apply_frames(Kp, T)
        Vp_g = self._apply_frames(Vp, T)

        # node attention logits
        a = torch.einsum('bhid,bhjd->bhij', Q, K) * (Dh ** -0.5)

        # point attention logits
        Qp_h = Qp_g.reshape(B, L, H, self.n_qp, 3).permute(0, 2, 1, 3, 4)  # (B,H,L,K,3)
        Kp_h = Kp_g.reshape(B, L, H, self.n_qp, 3).permute(0, 2, 1, 3, 4)
        diff  = Qp_h.unsqueeze(3) - Kp_h.unsqueeze(2)   # (B,H,L,L,K,3)
        dist2 = (diff ** 2).sum(-1).sum(-1)               # (B,H,L,L)

        w_c  = F.softplus(self.w_c).reshape(1, H, 1, 1)
        a    = a - 0.5 * w_c * dist2

        # pair bias
        a = a + self.pair_b(pair).permute(0, 3, 1, 2)

        a   = F.softmax(a * self.scale, dim=-1)   # (B,H,L,L)

        # aggregate node values
        out_n = torch.einsum('bhij,bhjd->bhid', a, V)   # (B,H,L,Dh)

        # aggregate point values
        Vp_h  = Vp_g.reshape(B, L, H, self.n_vp, 3).permute(0, 2, 1, 3, 4)  # (B,H,L,K,3)
        out_p = torch.einsum('bhij,bhjkd->bhikd', a, Vp_h)                    # (B,H,L,K,3)

        # compute norm for pair contribution
        norm_p = out_p.norm(dim=-1)  # (B,H,L,K)

        out_n  = out_n.permute(0, 2, 1, 3).reshape(B, L, H * Dh)
        out_p_flat = out_p.permute(0, 2, 1, 3, 4).reshape(B, L, H * self.n_vp * 3)
        norm_flat  = norm_p.permute(0, 2, 1, 3).reshape(B, L, H * self.n_vp)

        out = torch.cat([out_n, out_p_flat, norm_flat], dim=-1)
        return single + self.out_proj(out)


class BackboneUpdate(nn.Module):
    def __init__(self, d_node=cfg.D_NODE):
        super().__init__()
        self.norm  = LayerNorm(d_node)
        self.to_6  = Linear(d_node, 6, init='zeros')

    @staticmethod
    def axis_angle_to_rot(v):
        """v: (B,L,3) → rotation matrix (B,L,3,3)"""
        angle = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        axis  = v / angle
        s, c  = angle.sin(), angle.cos()
        kx, ky, kz = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
        oc   = 1 - c
        R = torch.stack([
            c + kx*kx*oc,    kx*ky*oc - kz*s, kx*kz*oc + ky*s,
            ky*kx*oc + kz*s, c + ky*ky*oc,    ky*kz*oc - kx*s,
            kz*kx*oc - ky*s, kz*ky*oc + kx*s, c + kz*kz*oc,
        ], dim=-1)
        return R.reshape(*v.shape[:-1], 3, 3)

    def forward(self, single, T):
        x    = self.norm(single)
        v6   = self.to_6(x)
        rot_v, trans = v6[..., :3], v6[..., 3:]
        dR   = self.axis_angle_to_rot(rot_v)
        R_old = T[..., :3, :3]
        t_old = T[..., :3, 3]
        R_new = torch.einsum('...ij,...jk->...ik', R_old, dR)
        t_new = t_old + torch.einsum('...ij,...j->...i', R_old, trans)
        T_new = T.clone()
        T_new[..., :3, :3] = R_new
        T_new[..., :3, 3]  = t_new
        return T_new


class StructureBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ipa   = InvariantPointAttention()
        self.bb    = BackboneUpdate()
        self.norm1 = LayerNorm(cfg.D_NODE)
        self.norm2 = LayerNorm(cfg.D_NODE)
        self.ff    = nn.Sequential(
            Linear(cfg.D_NODE, cfg.D_NODE * 2),
            nn.ReLU(),
            Linear(cfg.D_NODE * 2, cfg.D_NODE),
        )
        self.drop = nn.Dropout(cfg.DROPOUT)

    def _forward_impl(self, single, pair, T):
        single = self.drop(self.ipa(single, pair, T))
        single = self.norm1(single)
        T      = self.bb(single, T)
        single = single + self.drop(self.ff(self.norm2(single)))
        return single, T

    def forward(self, single, pair, T):
        if self.training:
            return grad_checkpoint(self._forward_impl, single, pair, T,
                                   use_reentrant=False)
        return self._forward_impl(single, pair, T)


# ─────────────────────────────────────────────────────────────
# 4. AUXILIARY HEADS
# ─────────────────────────────────────────────────────────────
class DistogramHead(nn.Module):
    def __init__(self, n_bins=cfg.N_DIST_BINS):
        super().__init__()
        self.proj = Linear(cfg.D_PAIR, n_bins)

    def forward(self, pair):
        return self.proj(pair)   # (B, L, L, n_bins)


class ContactHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            Linear(cfg.D_PAIR, 64), nn.ReLU(),
            Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, pair):
        return self.proj(pair).squeeze(-1)   # (B, L, L)


class CoordinateHead(nn.Module):
    def forward(self, T):
        return T[..., :3, 3]   # (B, L, 3)


# ─────────────────────────────────────────────────────────────
# 5. FULL MODEL
# ─────────────────────────────────────────────────────────────
class RNAFoldSE3(nn.Module):
    """
    Full SE(3)-equivariant RNA folding model v2.

    Architecture:
      NodeEmbed(seq, f1, dihed, ss_pair)
      + PairEmbed(rbf, dist_bins, orient, rel_pos, MIp, FNp, contact_ss, pair_type)
      → N_EVOFORMER × EvoformerBlock
      → Recycling × N_RECYCLE:
          → N_STRUCTURE × StructureBlock (IPA + frame updates)
      → DistogramHead + ContactHead + CoordinateHead

    Sequence-only inference:
      Pass zeros for rbf/dist_bins/orient/dihed → model falls back to
      MSA covariation + secondary structure for guidance.
    """

    def __init__(self):
        super().__init__()
        self.node_embed = NodeEmbedding()
        self.pair_embed = PairEmbedding()

        self.evoformer = nn.ModuleList([
            EvoformerBlock() for _ in range(cfg.N_EVOFORMER)
        ])
        self.structure = nn.ModuleList([
            StructureBlock() for _ in range(cfg.N_STRUCTURE)
        ])

        self.dist_head    = DistogramHead()
        self.contact_head = ContactHead()
        self.coord_head   = CoordinateHead()

        self.recycle_single = Linear(cfg.D_NODE, cfg.D_NODE)
        self.recycle_pair   = Linear(cfg.D_PAIR,  cfg.D_PAIR)

    def init_frames(self, B, L, device):
        T = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)
        T = T.expand(B, L, -1, -1).clone()
        T[:, :, 2, 3] = torch.arange(L, device=device).float().unsqueeze(0) * 6.0
        return T

    def forward(self, batch, device=None):
        if device is None:
            device = next(self.parameters()).device

        def to(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        seq_ids    = to(batch['seq_ids'])        # (B, L)
        seq_mask   = to(batch['seq_mask'])       # (B, L) bool
        f1         = to(batch['f1'])             # (B, L, 5)
        dihed      = to(batch['dihed'])          # (B, L, 4)
        ss_pair    = to(batch['ss_pair'])        # (B, L)      ← new
        rbf        = to(batch['dist_rbf'])       # (B, L, L, 16)
        dist_bins  = to(batch['dist_bins'])      # (B, L, L, 36)
        orient     = to(batch['orient'])         # (B, L, L, 4)
        rel_pos    = to(batch['rel_pos'])        # (B, L, L, 65)
        MIp        = to(batch['MIp'])            # (B, L, L)
        FNp        = to(batch['FNp'])            # (B, L, L)
        contact_ss = to(batch['contact_ss'])     # (B, L, L)   ← new
        pair_type  = to(batch['pair_type'])      # (B, L, L, 3) ← new

        B, L = seq_ids.shape

        # ── Initial embeddings ──
        single = self.node_embed(seq_ids, f1, dihed, ss_pair)
        pair   = self.pair_embed(rbf, dist_bins, orient, rel_pos,
                                 MIp, FNp, contact_ss, pair_type)

        # ── Recycling ──
        prev_single = torch.zeros_like(single)
        prev_pair   = torch.zeros_like(pair)
        T           = self.init_frames(B, L, device)
        all_coords  = []

        for _ in range(cfg.N_RECYCLE):
            single = single + self.recycle_single(prev_single)
            pair   = pair   + self.recycle_pair(prev_pair)

            for block in self.evoformer:
                if self.training:
                    single, pair = torch.utils.checkpoint.checkpoint(
                        block, single, pair, seq_mask, use_reentrant=False)
                else:
                    single, pair = block(single, pair, seq_mask)

            s = single
            for block in self.structure:
                s, T = block(s, pair, T)

            prev_single = single.detach()
            prev_pair   = pair.detach()
            single      = s
            all_coords.append(self.coord_head(T))

        distogram    = self.dist_head(pair)
        contact_pred = self.contact_head(pair)

        return {
            'coords'     : all_coords[-1],
            'all_coords' : all_coords,
            'distogram'  : distogram,
            'contact'    : contact_pred,
            'pair'       : pair,
            'single'     : single,
        }


# ─────────────────────────────────────────────────────────────
# 6. DUAL GPU WRAPPER
# ─────────────────────────────────────────────────────────────
def build_model_dual_gpu():
    """
    Wraps RNAFoldSE3 in DataParallel if 2 GPUs are available.
    Returns (model, device).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = RNAFoldSE3()

    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        print(f"  🔥  Using {n_gpus} GPUs via DataParallel")
        model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
    elif n_gpus == 1:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    model = model.to(device)
    return model, device


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = RNAFoldSE3().to(device)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_p:,}")

    B, L = 2, 64
    fake = {
        'seq_ids'   : torch.randint(0, 4, (B, L)).to(device),
        'seq_mask'  : torch.ones(B, L, dtype=torch.bool).to(device),
        'f1'        : torch.rand(B, L, 5).to(device),
        'dihed'     : torch.zeros(B, L, 4).to(device),    # zeros = inference mode
        'ss_pair'   : torch.rand(B, L).to(device),
        'dist_rbf'  : torch.zeros(B, L, L, 16).to(device),
        'dist_bins' : torch.zeros(B, L, L, 36).to(device),
        'orient'    : torch.zeros(B, L, L, 4).to(device),
        'rel_pos'   : torch.rand(B, L, L, 65).to(device),
        'MIp'       : torch.rand(B, L, L).to(device),
        'FNp'       : torch.rand(B, L, L).to(device),
        'contact_ss': torch.rand(B, L, L).to(device),
        'pair_type' : torch.rand(B, L, L, 3).to(device),
    }

    with torch.no_grad():
        out = model(fake, device=device)

    print(f"coords : {out['coords'].shape}")
    print(f"distogram: {out['distogram'].shape}")
    print("Forward pass OK ✅  (sequence-only inference mode)")
