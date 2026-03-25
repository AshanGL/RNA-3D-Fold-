"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SE(3)-EQUIVARIANT RNA FOLDING MODEL — KAGGLE WINNING LEVEL                 ║
║                                                                              ║
║  Key innovations:                                                            ║
║    1. Invariant Point Attention (IPA) — AlphaFold2-style                    ║
║    2. Pair-weighted residue update                                           ║
║    3. Structure module with frame updates (SE(3) equivariant)               ║
║    4. Recycling iterations (3 recycles)                                      ║
║    5. Auxiliary distance + contact heads for supervision                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
class ModelConfig:
    # Sequence / pair dims
    D_NODE   = 256     # single (node) representation dim
    D_PAIR   = 128     # pair representation dim
    D_HIDDEN = 128     # IPA hidden dim

    # Attention
    N_HEAD      = 8
    N_QUERY_PT  = 4    # IPA query points per head
    N_VALUE_PT  = 8    # IPA value points per head

    # Layers
    N_EVOFORMER = 8    # evoformer-style blocks
    N_STRUCTURE = 4    # structure module iterations
    N_RECYCLE   = 3    # recycling

    # Input feature dims (must match rna_features.py)
    N_DIST_BINS = 36
    N_ORIENT    = 4
    N_RBF       = 16
    N_REL_POS   = 65
    N_DIHED     = 4
    F1_DIM      = 5    # single-position frequency
    VOCAB_SIZE  = 6    # A U G C <PAD> <UNK>

    DROPOUT     = 0.1
    MAX_LEN     = 512

cfg = ModelConfig()

# ─────────────────────────────────────────────────────────────
# UTILITY LAYERS
# ─────────────────────────────────────────────────────────────
class LayerNorm(nn.LayerNorm):
    """LayerNorm with float32 upcast for stability."""
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


class Linear(nn.Linear):
    """Linear with optional init mode."""
    def __init__(self, in_f, out_f, bias=True, init='default'):
        super().__init__(in_f, out_f, bias=bias)
        if init == 'relu':
            nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        elif init == 'zeros':
            nn.init.zeros_(self.weight)
            if bias: nn.init.zeros_(self.bias)
        elif init == 'final':
            nn.init.zeros_(self.weight)
            if bias: nn.init.zeros_(self.bias)
        else:
            nn.init.xavier_uniform_(self.weight)
            if bias: nn.init.zeros_(self.bias)


def exists(x): return x is not None


# ─────────────────────────────────────────────────────────────
# 1. INPUT EMBEDDINGS
# ─────────────────────────────────────────────────────────────
class NodeEmbedding(nn.Module):
    """
    Builds single (node) representation from:
      - sequence token embedding
      - f1 frequency profile (5 dims)
      - dihedral angles (4 dims)
      - sinusoidal positional encoding
    """
    def __init__(self):
        super().__init__()
        self.seq_embed  = nn.Embedding(cfg.VOCAB_SIZE, 64, padding_idx=4)
        self.f1_proj    = Linear(cfg.F1_DIM, 32)
        self.dihed_proj = Linear(cfg.N_DIHED, 32)
        self.pe         = SinusoidalPE(cfg.D_NODE, cfg.MAX_LEN)

        in_dim = 64 + 32 + 32
        self.out_proj = nn.Sequential(
            Linear(in_dim, cfg.D_NODE),
            nn.ReLU(),
            LayerNorm(cfg.D_NODE),
        )

    def forward(self, seq_ids, f1, dihed):
        s = self.seq_embed(seq_ids)         # (B, L, 64)
        f = self.f1_proj(f1)               # (B, L, 32)
        d = self.dihed_proj(dihed)         # (B, L, 32)
        x = torch.cat([s, f, d], dim=-1)   # (B, L, 128)
        x = self.out_proj(x)               # (B, L, D_NODE)
        x = self.pe(x)
        return x


class PairEmbedding(nn.Module):
    """
    Builds pair representation from:
      - RBF distances (16 dims)
      - binned distances (36 dims)
      - orientation (4 dims)
      - relative position (65 dims)
      - MIp covariation (1 dim)
      - FNp direct information (1 dim)
    """
    def __init__(self):
        super().__init__()
        in_dim = cfg.N_RBF + cfg.N_DIST_BINS + cfg.N_ORIENT + cfg.N_REL_POS + 2
        self.proj = nn.Sequential(
            Linear(in_dim, cfg.D_PAIR * 2),
            nn.ReLU(),
            Linear(cfg.D_PAIR * 2, cfg.D_PAIR),
            LayerNorm(cfg.D_PAIR),
        )

    def forward(self, rbf, dist_bins, orient, rel_pos, MIp, FNp):
        """
        rbf       : (B, L, L, 16)
        dist_bins : (B, L, L, 36)
        orient    : (B, L, L, 4)
        rel_pos   : (B, L, L, 65)
        MIp       : (B, L, L)
        FNp       : (B, L, L)
        """
        cov = torch.stack([MIp, FNp], dim=-1)  # (B, L, L, 2)
        x = torch.cat([rbf, dist_bins, orient, rel_pos, cov], dim=-1)
        return self.proj(x)   # (B, L, L, D_PAIR)


class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ─────────────────────────────────────────────────────────────
# 2. EVOFORMER-STYLE BLOCKS
# ─────────────────────────────────────────────────────────────
class RowAttentionWithPairBias(nn.Module):
    """
    Row-wise gated multi-head self-attention with pair bias.
    single: (B, L, D_NODE)
    pair  : (B, L, L, D_PAIR)
    """
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
        H, Dh = self.n_head, self.d_head

        x  = self.norm(single)
        Q  = self.q(x).reshape(B, L, H, Dh).permute(0, 2, 1, 3)   # (B,H,L,Dh)
        K  = self.k(x).reshape(B, L, H, Dh).permute(0, 2, 1, 3)
        V  = self.v(x).reshape(B, L, H, Dh).permute(0, 2, 1, 3)
        g  = torch.sigmoid(self.gate(x)).reshape(B, L, H, Dh).permute(0, 2, 1, 3)

        attn  = torch.einsum('bhid,bhjd->bhij', Q, K) * self.scale   # (B,H,L,L)
        bias  = self.pair_bias(pair).permute(0, 3, 1, 2)              # (B,H,L,L)
        attn  = attn + bias

        if mask is not None:
            # mask: (B, L)  → (B, 1, 1, L)
            attn = attn.masked_fill(~mask[:, None, None, :], -1e9)

        attn  = F.softmax(attn, dim=-1)
        out   = torch.einsum('bhij,bhjd->bhid', attn, V)              # (B,H,L,Dh)
        out   = (g * out).permute(0, 2, 1, 3).reshape(B, L, D)
        return single + self.out(out)


class PairUpdate(nn.Module):
    """
    Update pair representation using outer product of single.
    single: (B, L, D) → pair update: (B, L, L, D_PAIR)
    """
    def __init__(self, d_node=cfg.D_NODE, d_pair=cfg.D_PAIR):
        super().__init__()
        self.norm  = LayerNorm(d_node)
        self.proj_a = Linear(d_node, d_pair)
        self.proj_b = Linear(d_node, d_pair)
        self.out    = Linear(d_pair, d_pair, init='final')
        self.pair_norm = LayerNorm(d_pair)

    def forward(self, single, pair):
        x = self.norm(single)
        a = self.proj_a(x)   # (B, L, D_PAIR)
        b = self.proj_b(x)

        # outer product mean
        outer = torch.einsum('bid,bjd->bijd', a, b)  # (B, L, L, D_PAIR)
        pair  = self.pair_norm(pair + self.out(outer))
        return pair


class TriangleAttention(nn.Module):
    """
    Triangle self-attention (starting node / ending node).
    Pair: (B, L, L, D_PAIR)
    """
    def __init__(self, d_pair=cfg.D_PAIR, n_head=4, mode='start'):
        super().__init__()
        assert mode in ('start', 'end')
        self.mode   = mode
        self.n_head = n_head
        self.d_head = d_pair // n_head
        self.scale  = self.d_head ** -0.5

        self.norm  = LayerNorm(d_pair)
        self.q     = Linear(d_pair, d_pair, bias=False)
        self.k     = Linear(d_pair, d_pair, bias=False)
        self.v     = Linear(d_pair, d_pair, bias=False)
        self.gate  = Linear(d_pair, d_pair)
        self.out   = Linear(d_pair, d_pair, init='final')
        self.bias  = Linear(d_pair, n_head, bias=False)

    def forward(self, pair):
        """pair: (B, L, L, D_PAIR)"""
        if self.mode == 'end':
            pair = pair.permute(0, 2, 1, 3)   # swap rows/cols

        B, L, _, D = pair.shape
        H, Dh = self.n_head, self.d_head

        x = self.norm(pair)
        Q = self.q(x).reshape(B, L, L, H, Dh).permute(0, 1, 3, 2, 4)  # (B,L,H,L,Dh)
        K = self.k(x).reshape(B, L, L, H, Dh).permute(0, 1, 3, 2, 4)
        V = self.v(x).reshape(B, L, L, H, Dh).permute(0, 1, 3, 2, 4)
        g = torch.sigmoid(self.gate(x)).reshape(B, L, L, H, Dh).permute(0, 1, 3, 2, 4)

        attn = torch.einsum('blhid,blhjd->blhij', Q, K) * self.scale
        bias = self.bias(pair).permute(0, 1, 3, 2).unsqueeze(3)  # (B,L,H,1,L)
        attn = F.softmax(attn + bias, dim=-1)

        out = torch.einsum('blhij,blhjd->blhid', attn, V)        # (B,L,H,L,Dh)
        out = (g * out).permute(0, 1, 3, 2, 4).reshape(B, L, L, D)
        result = pair + self.out(out)

        if self.mode == 'end':
            result = result.permute(0, 2, 1, 3)
        return result


class TriangleMultiplication(nn.Module):
    """Triangular multiplicative update (outgoing or incoming)."""
    def __init__(self, d_pair=cfg.D_PAIR, mode='outgoing'):
        super().__init__()
        assert mode in ('outgoing', 'incoming')
        self.mode = mode
        self.norm   = LayerNorm(d_pair)
        self.left_a = Linear(d_pair, d_pair)
        self.left_b = Linear(d_pair, d_pair)
        self.right_a = Linear(d_pair, d_pair)
        self.right_b = Linear(d_pair, d_pair)
        self.gate_a = Linear(d_pair, d_pair)
        self.gate_b = Linear(d_pair, d_pair)
        self.norm_out = LayerNorm(d_pair)
        self.out    = Linear(d_pair, d_pair, init='final')
        self.gate   = Linear(d_pair, d_pair)

    def forward(self, pair):
        x = self.norm(pair)   # (B, L, L, D)
        if self.mode == 'outgoing':
            # p(i,j) += Σ_k la(i,k) * lb(j,k)
            la = torch.sigmoid(self.gate_a(x)) * self.left_a(x)   # (B,L,L,D)
            lb = torch.sigmoid(self.gate_b(x)) * self.left_b(x)
            p  = torch.einsum('bikd,bjkd->bijd', la, lb)
        else:
            # p(i,j) += Σ_k ra(k,i) * rb(k,j)
            ra = torch.sigmoid(self.gate_a(x)) * self.right_a(x)
            rb = torch.sigmoid(self.gate_b(x)) * self.right_b(x)
            p  = torch.einsum('bkid,bkjd->bijd', ra, rb)

        g   = torch.sigmoid(self.gate(x))
        out = g * self.out(self.norm_out(p))
        return pair + out


class EvoformerBlock(nn.Module):
    """One Evoformer block: row attn → pair update → tri-mult → tri-attn."""
    def __init__(self):
        super().__init__()
        self.row_attn  = RowAttentionWithPairBias()
        self.pair_upd  = PairUpdate()
        self.tri_out   = TriangleMultiplication(mode='outgoing')
        self.tri_in    = TriangleMultiplication(mode='incoming')
        self.tri_start = TriangleAttention(mode='start')
        self.tri_end   = TriangleAttention(mode='end')
        self.ff_single = nn.Sequential(
            LayerNorm(cfg.D_NODE),
            Linear(cfg.D_NODE, cfg.D_NODE * 4),
            nn.ReLU(),
            Linear(cfg.D_NODE * 4, cfg.D_NODE, init='final'),
        )
        self.ff_pair = nn.Sequential(
            LayerNorm(cfg.D_PAIR),
            Linear(cfg.D_PAIR, cfg.D_PAIR * 4),
            nn.ReLU(),
            Linear(cfg.D_PAIR * 4, cfg.D_PAIR, init='final'),
        )
        self.drop = nn.Dropout(cfg.DROPOUT)

    def forward(self, single, pair, mask=None):
        single = self.row_attn(single, pair, mask)
        single = single + self.drop(self.ff_single(single))

        pair   = self.pair_upd(single, pair)
        pair   = self.tri_out(pair)
        pair   = self.tri_in(pair)
        pair   = self.tri_start(pair)
        pair   = self.tri_end(pair)
        pair   = pair + self.drop(self.ff_pair(pair))
        return single, pair


# ─────────────────────────────────────────────────────────────
# 3. SE(3)-EQUIVARIANT STRUCTURE MODULE
# ─────────────────────────────────────────────────────────────
class InvariantPointAttention(nn.Module):
    """
    IPA as described in AlphaFold2, adapted for RNA.
    Inputs:
      single : (B, L, D_NODE)
      pair   : (B, L, L, D_PAIR)
      T      : (B, L, 4, 4) rigid frames (rotation + translation)
    Output:
      updated single: (B, L, D_NODE)
    """
    def __init__(self,
                 d_node   = cfg.D_NODE,
                 d_pair   = cfg.D_PAIR,
                 d_hidden = cfg.D_HIDDEN,
                 n_head   = cfg.N_HEAD,
                 n_qp     = cfg.N_QUERY_PT,
                 n_vp     = cfg.N_VALUE_PT):
        super().__init__()
        self.n_head   = n_head
        self.n_qp     = n_qp
        self.n_vp     = n_vp
        self.d_head   = d_hidden // n_head
        self.scale    = (self.d_head + n_qp * 9) ** -0.5
        self.w_c      = (2.0 / (9 * n_qp)) ** 0.5
        self.w_l      = (1.0 / 3.0) ** 0.5

        # standard q/k/v projections
        self.to_q  = Linear(d_node, d_hidden * n_head, bias=False)
        self.to_k  = Linear(d_node, d_hidden * n_head, bias=False)
        self.to_v  = Linear(d_node, d_hidden * n_head, bias=False)

        # point projections (3D)
        self.to_qp = Linear(d_node, n_head * n_qp * 3, bias=False)
        self.to_kp = Linear(d_node, n_head * n_qp * 3, bias=False)
        self.to_vp = Linear(d_node, n_head * n_vp * 3, bias=False)

        # pair bias
        self.pair_bias = Linear(d_pair, n_head, bias=False)

        # scalar head weight (learnable)
        self.head_w = nn.Parameter(torch.zeros(n_head))

        # output
        out_dim = n_head * (d_hidden + n_vp * 3 + n_vp + d_pair)
        self.out_proj = Linear(out_dim, d_node, init='final')
        self.norm = LayerNorm(d_node)

    @staticmethod
    def apply_rotation(R, pts):
        """
        R   : (B, L, 3, 3)
        pts : (B, L, n, 3)
        → (B, L, n, 3)
        """
        return torch.einsum('blij,blnj->blni', R, pts)

    @staticmethod
    def apply_rigid(T, pts):
        """
        T   : (B, L, 4, 4)
        pts : (B, L, n, 3)
        → apply [R|t] to points
        """
        R = T[..., :3, :3]
        t = T[..., :3, 3].unsqueeze(-2)   # (B, L, 1, 3)
        return torch.einsum('blij,blnj->blni', R, pts) + t

    def forward(self, single, pair, T):
        B, L, D = single.shape
        H, Dh, Nqp, Nvp = self.n_head, self.d_head, self.n_qp, self.n_vp
        x = self.norm(single)

        # ── standard attention ──
        Q = self.to_q(x).reshape(B, L, H, Dh)
        K = self.to_k(x).reshape(B, L, H, Dh)
        V = self.to_v(x).reshape(B, L, H, Dh)

        # ── point queries/keys/values in local frame ──
        Qp = self.to_qp(x).reshape(B, L, H, Nqp, 3)   # local frame
        Kp = self.to_kp(x).reshape(B, L, H, Nqp, 3)
        Vp = self.to_vp(x).reshape(B, L, H, Nvp, 3)

        R  = T[..., :3, :3]   # (B, L, 3, 3)

        # rotate point queries/keys to global frame per residue
        Qp_g = torch.einsum('blij,blhqj->blhqi', R, Qp)   # (B,L,H,Nqp,3)
        Kp_g = torch.einsum('blij,blhqj->blhqi', R, Kp)
        Vp_g = torch.einsum('blij,blhqj->blhvi', R, Vp)

        # ── attention logits ──
        # scalar part: (B, H, L, L)
        attn_s = torch.einsum('blhd,bmhd->bhlm', Q, K) * self.scale

        # point part: sum over query points of squared distances
        # (B, H, L, L)  — distance² between Q-points of i and K-points of j
        diffs = Qp_g.unsqueeze(3) - Kp_g.unsqueeze(2)    # (B,L,L,H,Nqp,3)
        dist2 = (diffs ** 2).sum(-1).sum(-1)               # (B,L,L,H)
        attn_p = -0.5 * self.w_c * dist2.permute(0, 3, 1, 2)  # (B,H,L,L)

        # pair bias
        pb = self.pair_bias(pair).permute(0, 3, 1, 2)     # (B,H,L,L)

        # head weights
        hw = F.softplus(self.head_w).reshape(1, H, 1, 1)
        attn = F.softmax(hw * (attn_s + attn_p) + pb, dim=-1)   # (B,H,L,L)

        # ── aggregate ──
        # scalar
        out_s = torch.einsum('bhlm,bmhd->blhd', attn, V)          # (B,L,H,Dh)

        # points (global frame)
        out_p = torch.einsum('bhlm,bmhvi->blhvi', attn, Vp_g)     # (B,L,H,Nvp,3)
        # rotate back to local frame
        R_inv = R.transpose(-1, -2)
        out_p_local = torch.einsum('blij,blhvj->blhvi', R_inv, out_p)  # (B,L,H,Nvp,3)
        out_p_norm  = out_p_local.norm(dim=-1)                          # (B,L,H,Nvp)

        # pair features at each attended pair
        out_pair = torch.einsum('bhlm,blmd->blhd', attn, pair)   # (B,L,H,D_PAIR)

        # concat all
        out = torch.cat([
            out_s.reshape(B, L, H * Dh),
            out_p_local.reshape(B, L, H * Nvp * 3),
            out_p_norm.reshape(B, L, H * Nvp),
            out_pair.reshape(B, L, H * pair.shape[-1]),
        ], dim=-1)    # (B, L, out_dim)

        return single + self.out_proj(out)


class BackboneUpdate(nn.Module):
    """
    Predict Δ(rotation, translation) for each residue's rigid frame.
    Updates T: (B, L, 4, 4)
    """
    def __init__(self, d_node=cfg.D_NODE):
        super().__init__()
        self.norm  = LayerNorm(d_node)
        self.to_6  = Linear(d_node, 6)   # 3 for rotation (axis-angle) + 3 for translation

    @staticmethod
    def axis_angle_to_rot(v):
        """
        v: (B, L, 3) small rotation vector
        Returns (B, L, 3, 3) rotation matrix via Rodrigues formula.
        """
        angle = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # (B,L,1)
        axis  = v / angle                                        # (B,L,3)
        a     = angle.squeeze(-1)                               # (B,L)
        x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]

        cos_a = torch.cos(a).unsqueeze(-1).unsqueeze(-1)        # (B,L,1,1)
        sin_a = torch.sin(a).unsqueeze(-1).unsqueeze(-1)

        # K = cross-product matrix
        z_ = torch.zeros_like(x)
        K = torch.stack([
            torch.stack([ z_, -z,  y], dim=-1),
            torch.stack([ z,  z_, -x], dim=-1),
            torch.stack([-y,   x, z_], dim=-1),
        ], dim=-2)   # (B,L,3,3)

        I = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(K)
        R = cos_a * I + sin_a * K + (1 - cos_a) * torch.einsum('...i,...j->...ij', axis, axis)
        return R

    def forward(self, single, T):
        """T: (B,L,4,4) rigid frames."""
        x  = self.norm(single)
        v6 = self.to_6(x)                    # (B,L,6)
        rot_v, trans = v6[..., :3], v6[..., 3:]

        dR = self.axis_angle_to_rot(rot_v)   # (B,L,3,3)
        dt = trans                           # (B,L,3)

        # compose with existing frame
        R_old = T[..., :3, :3]              # (B,L,3,3)
        t_old = T[..., :3, 3]              # (B,L,3)

        R_new = torch.einsum('...ij,...jk->...ik', R_old, dR)
        t_new = t_old + torch.einsum('...ij,...j->...i', R_old, dt)

        T_new = T.clone()
        T_new[..., :3, :3] = R_new
        T_new[..., :3, 3]  = t_new
        return T_new


class StructureBlock(nn.Module):
    """One Structure Module block: IPA → backbone update → feedforward."""
    def __init__(self):
        super().__init__()
        self.ipa  = InvariantPointAttention()
        self.bb   = BackboneUpdate()
        self.norm1 = LayerNorm(cfg.D_NODE)
        self.norm2 = LayerNorm(cfg.D_NODE)
        self.ff   = nn.Sequential(
            Linear(cfg.D_NODE, cfg.D_NODE * 2),
            nn.ReLU(),
            Linear(cfg.D_NODE * 2, cfg.D_NODE),
        )
        self.drop = nn.Dropout(cfg.DROPOUT)

    def forward(self, single, pair, T):
        single = self.drop(self.ipa(single, pair, T))
        single = self.norm1(single)
        T      = self.bb(single, T)
        single = self.drop(self.ff(single))
        single = self.norm2(single)
        return single, T


# ─────────────────────────────────────────────────────────────
# 4. AUXILIARY HEADS
# ─────────────────────────────────────────────────────────────
class DistogramHead(nn.Module):
    """Predict binned distance distribution from pair repr."""
    def __init__(self, n_bins=cfg.N_DIST_BINS):
        super().__init__()
        self.proj = Linear(cfg.D_PAIR, n_bins)

    def forward(self, pair):
        return self.proj(pair)   # (B, L, L, n_bins)


class ContactHead(nn.Module):
    """Binary contact prediction from pair repr."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            Linear(cfg.D_PAIR, 64), nn.ReLU(),
            Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, pair):
        return self.proj(pair).squeeze(-1)   # (B, L, L)


class CoordinateHead(nn.Module):
    """Extract C1' xyz from final rigid frames."""
    def forward(self, T):
        """T: (B, L, 4, 4) → (B, L, 3)"""
        return T[..., :3, 3]


# ─────────────────────────────────────────────────────────────
# 5. FULL MODEL
# ─────────────────────────────────────────────────────────────
class RNAFoldSE3(nn.Module):
    """
    Full SE(3)-equivariant RNA folding model.
    Architecture:
      NodeEmbed + PairEmbed
      → N_EVOFORMER × EvoformerBlock        (refines single + pair)
      → Recycling loop (N_RECYCLE times):
          → N_STRUCTURE × StructureBlock    (IPA + frame updates)
      → DistogramHead + ContactHead + CoordinateHead
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

        # recycling projections
        self.recycle_single = Linear(cfg.D_NODE, cfg.D_NODE)
        self.recycle_pair   = Linear(cfg.D_PAIR,  cfg.D_PAIR)

    def init_frames(self, B, L, device):
        """Initialize identity rigid frames (B, L, 4, 4)."""
        T = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)
        T = T.expand(B, L, -1, -1).clone()
        # init translations with sequential positions
        T[:, :, 2, 3] = torch.arange(L, device=device).float().unsqueeze(0) * 6.0
        return T

    def forward(self, batch, device=None):
        if device is None:
            device = next(self.parameters()).device

        def to(x): return x.to(device) if isinstance(x, torch.Tensor) else x

        seq_ids   = to(batch['seq_ids'])      # (B, L)
        seq_mask  = to(batch['seq_mask'])     # (B, L) bool
        f1        = to(batch['f1'])           # (B, L, 5)
        dihed     = to(batch['dihed'])        # (B, L, 4)
        rbf       = to(batch['dist_rbf'])     # (B, L, L, 16)
        dist_bins = to(batch['dist_bins'])    # (B, L, L, 36)
        orient    = to(batch['orient'])       # (B, L, L, 4)
        rel_pos   = to(batch['rel_pos'])      # (B, L, L, 65)
        MIp       = to(batch['MIp'])          # (B, L, L)
        FNp       = to(batch['FNp'])          # (B, L, L)

        B, L = seq_ids.shape

        # ── Initial embeddings ──
        single = self.node_embed(seq_ids, f1, dihed)       # (B, L, D_NODE)
        pair   = self.pair_embed(rbf, dist_bins, orient,
                                 rel_pos, MIp, FNp)        # (B, L, L, D_PAIR)

        # ── Recycling ──
        prev_single = torch.zeros_like(single)
        prev_pair   = torch.zeros_like(pair)
        T           = self.init_frames(B, L, device)

        all_coords = []

        for recycle in range(cfg.N_RECYCLE):
            # add recycled representation
            single = single + self.recycle_single(prev_single)
            pair   = pair   + self.recycle_pair(prev_pair)

            # ── Evoformer stack ──
            for block in self.evoformer:
                single, pair = block(single, pair, seq_mask)

            # ── Structure module ──
            s = single
            for block in self.structure:
                s, T = block(s, pair, T)

            prev_single = single.detach()
            prev_pair   = pair.detach()
            single      = s

            coords = self.coord_head(T)   # (B, L, 3)
            all_coords.append(coords)

        # ── Auxiliary outputs ──
        distogram    = self.dist_head(pair)    # (B, L, L, n_bins)
        contact_pred = self.contact_head(pair) # (B, L, L)

        return {
            'coords'     : all_coords[-1],      # (B, L, 3)  — final
            'all_coords' : all_coords,           # list of (B,L,3) per recycle
            'distogram'  : distogram,
            'contact'    : contact_pred,
            'pair'       : pair,
            'single'     : single,
        }


# ─────────────────────────────────────────────────────────────
# 6. QUICK MODEL SUMMARY
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = RNAFoldSE3().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # synthetic forward pass
    B, L = 2, 64
    fake_batch = {
        'seq_ids'  : torch.randint(0, 4, (B, L)).to(device),
        'seq_mask' : torch.ones(B, L, dtype=torch.bool).to(device),
        'f1'       : torch.rand(B, L, 5).to(device),
        'dihed'    : torch.rand(B, L, 4).to(device),
        'dist_rbf' : torch.rand(B, L, L, 16).to(device),
        'dist_bins': torch.rand(B, L, L, 36).to(device),
        'orient'   : torch.rand(B, L, L, 4).to(device),
        'rel_pos'  : torch.rand(B, L, L, 65).to(device),
        'MIp'      : torch.rand(B, L, L).to(device),
        'FNp'      : torch.rand(B, L, L).to(device),
    }

    with torch.no_grad():
        out = model(fake_batch, device=device)

    print(f"coords shape   : {out['coords'].shape}")
    print(f"distogram shape: {out['distogram'].shape}")
    print(f"contact shape  : {out['contact'].shape}")
    print("Forward pass OK ✅")
