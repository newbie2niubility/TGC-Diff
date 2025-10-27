import math
import logging
from random import random
from functools import partial
from pathlib import Path

import torch
from torch import nn, Tensor, einsum, IntTensor, FloatTensor, BoolTensor
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack

from beartype import beartype
from beartype.typing import Tuple, Optional, List, Union
from model.attend import Attend

from abc import ABC
from tqdm import tqdm


def divisible_by(num, den):
    return (num % den) == 0

def default(val, d):
    return val if exists(val) else d

def exists(val):
    return val is not None

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def coin_flip():
    return random() < 0.5  
# sinusoidal positions

class LearnedSinusoidalPosEmb(Module):
    """ used by @crowsonkb """

    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        return fouriered


class RotaryEmbedding(Module):
    def __init__(self, dim, theta = 50000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @autocast(device_type='cuda', enabled=False)
    @beartype
    def forward(self, t: Union[int, Tensor]):
        if not torch.is_tensor(t):
            t = torch.arange(t, device = self.device)

        t = t.type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

@autocast(device_type='cuda', enabled = False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()


# convolutional positional generating module

class ConvPositionEmbed(Module):
    def __init__(
        self,
        dim,
        *,
        kernel_size,
        groups = None
    ):
        super().__init__()
        assert is_odd(kernel_size)
        groups = default(groups, dim) # full depthwise conv by default

        self.dw_conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups = groups, padding = kernel_size // 2),
            nn.GELU()
        )

    def forward(self, x, mask = None):

        if exists(mask):
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.)

        x = rearrange(x, 'b n c -> b c n')
        x = self.dw_conv1d(x)
        out = rearrange(x, 'b c n -> b n c')

        if exists(mask):
            out = out.masked_fill(~mask, 0.)

        return out

class RMSNorm(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        cond_dim = None
    ):
        super().__init__()
        cond_dim = default(cond_dim, dim)
        self.scale = dim ** 0.5

        self.to_gamma = nn.Linear(cond_dim, dim)
        self.to_beta = nn.Linear(cond_dim, dim)

        # init to identity

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)

        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, *, cond):
        normed = F.normalize(x, dim = -1) * self.scale

        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (gamma, beta))

        return normed * gamma + beta


class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0,
        flash = False,
        qk_norm = False,
        qk_norm_scale = 10
    ):
        super().__init__()
        self.heads = heads
        dim_inner = dim_head * heads

        scale = qk_norm_scale if qk_norm else None

        self.attend = Attend(dropout, flash = flash, scale = scale)

        self.qk_norm = qk_norm

        if qk_norm:
            self.q_norm = MultiheadRMSNorm(dim_head, heads = heads)
            self.k_norm = MultiheadRMSNorm(dim_head, heads = heads)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x, mask = None, rotary_emb = None):
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x


def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
    )

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        num_register_tokens = 0.,
        attn_flash = False,
        adaptive_rmsnorm = True,
        adaptive_rmsnorm_cond_dim_in = 4 * 512,
        use_unet_skip_connection = False,
        skip_connect_scale = None,
        attn_qk_norm = False,
        use_gateloop_layers = False,
        gateloop_use_jax = False,
    ):
        super().__init__()
        assert divisible_by(depth, 2)
        self.layers = nn.ModuleList([])

        self.rotary_emb = RotaryEmbedding(dim = dim_head)

        self.num_register_tokens = num_register_tokens
        self.has_register_tokens = num_register_tokens > 0

        if self.has_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        if adaptive_rmsnorm:
            rmsnorm_klass = partial(AdaptiveRMSNorm, cond_dim = adaptive_rmsnorm_cond_dim_in)
        else:
            rmsnorm_klass = RMSNorm

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                GateLoop(dim = dim, use_jax_associative_scan = gateloop_use_jax, post_ln = True) if use_gateloop_layers else None,
                rmsnorm_klass(dim = dim),
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = attn_flash, qk_norm = attn_qk_norm),
                rmsnorm_klass(dim = dim),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.final_norm = RMSNorm(dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        mask = None,
        adaptive_rmsnorm_cond = None
    ):
        batch, seq_len, *_ = x.shape

        # add register tokens to the left

        if self.has_register_tokens:
            register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = batch)

            x, ps = pack([register_tokens, x], 'b * d')

            if exists(mask):
                mask = F.pad(mask, (self.num_register_tokens, 0), value = True)

        # keep track of skip connections

        skip_connects = []

        # rotary embeddings

        positions = seq_len

        if self.has_register_tokens:
            main_positions = torch.arange(seq_len, device = self.device, dtype = torch.long)
            register_positions = torch.full((self.num_register_tokens,), -10000, device = self.device, dtype = torch.long)
            positions = torch.cat((register_positions, main_positions))

        rotary_emb = self.rotary_emb(positions)

        # adaptive rmsnorm

        rmsnorm_kwargs = dict()
        if exists(adaptive_rmsnorm_cond):
            rmsnorm_kwargs = dict(cond = adaptive_rmsnorm_cond)

        # going through the attention layers

        for maybe_gateloop, attn_prenorm, attn, ff_prenorm, ff in self.layers:
            if exists(maybe_gateloop):
                x = maybe_gateloop(x) + x

            attn_input = attn_prenorm(x, **rmsnorm_kwargs)
            x = attn(attn_input, mask = mask, rotary_emb = rotary_emb) + x

            ff_input = ff_prenorm(x, **rmsnorm_kwargs) 
            x = ff(ff_input) + x

        # remove the register tokens

        if self.has_register_tokens:
            _, x = unpack(x, ps, 'b * d')

        return self.final_norm(x)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob


def mask_from_start_end_indices(
    seq_len: int,
    start: Tensor,
    end: Tensor
):
    assert start.shape == end.shape
    device = start.device

    seq = torch.arange(seq_len, device = device, dtype = torch.long)
    seq = seq.reshape(*((-1,) * start.ndim), seq_len)
    seq = seq.expand(*start.shape, seq_len)

    mask = seq >= start[..., None].long()
    mask &= seq < end[..., None].long()
    return mask

def mask_from_frac_lengths(
    seq_len: int,
    frac_lengths: Tensor
):
    device = frac_lengths.device

    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.zeros_like(frac_lengths, device = device).float().uniform_(0, 1)
    start = (max_start * rand).clamp(min = 0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


class Temp(Module):
    @beartype
    def __init__(
        self,
        *,
        dim_in = 4, 
        dim_out = 4,
        vocab_size = None,
        dim_char_emb = 256,
        dim_line_emb = 256,
        dim = 512,
        depth = 10,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        ff_dropout = 0.,
        conv_pos_embed_kernel_size = 31,
        conv_pos_embed_groups = None,
        attn_dropout = 0,
        attn_flash = False,
        attn_qk_norm = True,
        use_gateloop_layers = False,
        p_drop_prob = 0.2, # p_drop in paper
        frac_lengths_mask: Tuple[float, float] = (0.1, 1.),
        aligner_kwargs: dict = dict(dim_in = 80, attn_channels = 80)
    ):
        super().__init__()

        self.proj_in = nn.Linear(dim_in, dim)

        self.token_emb = nn.Embedding(vocab_size, dim_char_emb)

        self.p_drop_prob = p_drop_prob
        self.frac_lengths_mask = frac_lengths_mask

        self.to_embed = nn.Linear(dim*2 + dim_char_emb, dim)

        self.conv_embed = ConvPositionEmbed(
            dim = dim,
            kernel_size = conv_pos_embed_kernel_size,
            groups = conv_pos_embed_groups
        )

        self.sinu_pos_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU()
        )
        
        
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout,
            attn_dropout=attn_dropout,
            attn_flash = attn_flash,
            attn_qk_norm = attn_qk_norm,
            use_gateloop_layers = use_gateloop_layers
        )

        self.to_pred = nn.Sequential(
            nn.Linear(dim, dim * 2),
            GEGLU(),
            nn.Linear(dim, dim_out)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @beartype
    def forward(
        self,
        *,
        x_t,
        times,
        cond,
        char_ids = None,
        target = None,
        cond_mask = None,
        pred_mask = None,
    ):
        batch, seq_len, cond_dim = cond.shape
        
        self_attn_mask = pred_mask + cond_mask
        self_attn_mask = self_attn_mask.to(torch.bool)
        cond = self.proj_in(cond)
        x_t = self.proj_in(x_t)

        cond = cond * cond_mask.unsqueeze(dim=-1)

        char_emb = self.token_emb(char_ids)

        # combine audio, phoneme, conditioning
        embed = torch.cat((x_t, cond, char_emb), dim = -1)

        x = self.to_embed(embed)

        x = self.conv_embed(x, mask = self_attn_mask) + x

        time_emb = self.sinu_pos_emb(times)

        x = self.transformer(
            x,
            mask = self_attn_mask,
            adaptive_rmsnorm_cond = time_emb
        )
        locations = self.to_pred(x)

        if not self.training:
            return locations

        loss = F.mse_loss(locations, target, reduction = 'none')
        loss = loss  * pred_mask.unsqueeze(dim=-1)
        # masked mean
        num = reduce(loss, 'b n c-> b c', 'sum')
        den = pred_mask.sum(dim = -1).clamp(min = 1e-5)
        loss = num / den.unsqueeze(-1)
        loss = loss.mean(dim=0)

        return loss

class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
    ):
        super().__init__()
        self.estimator = None
        self.in_channels = 4

    @torch.inference_mode()
    def inference(self, 
            cond,
            char_ids = None,
            target = None,
            cond_mask = None,
            pred_mask = None,
            n_timesteps = None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=target.device, dtype=target.dtype)
        x = torch.randn_like(target) 
        _, t, _ = t_span[0], t_span[-1], t_span[1] - t_span[0]
        
        B, T, C = target.shape
        t = t.unsqueeze(0).repeat(B)
        sol = []
        
        for step in tqdm(range(1, len(t_span))):
            dt = t_span[step] - t_span[step - 1]
            # import pdb;pdb.set_trace()
            dphi_dt = self.estimator(x_t = x, times = t, cond = cond, char_ids = char_ids , target = None, cond_mask=cond_mask, pred_mask=pred_mask) 
            x = x - dt * dphi_dt
            t = t - dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return sol[-1]

    def solve_euler(self, x, target, content_emb, width, images_ids, txt_id, t_span, inference_cfg_rate=0.5):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        

    def forward(
            self,
            cond,
            target,
            char_ids,
            pred_mask,
            cond_mask
        ):
        x_0 = cond
        device = x_0.device
        dtype = x_0.dtype
        bs = x_0.shape[0]
        t = torch.sigmoid(torch.randn((bs,), device=device)).to(dtype)
        x_1 = torch.randn_like(x_0).to(device)
        t_ = t.unsqueeze(1).unsqueeze(1)
        x_t = ((1 - t_) * x_0 + t_ * x_1).to(dtype)
        u = (x_1 - x_0).to(dtype)

        loss = self.estimator(x_t = x_t, times = t, cond = cond, char_ids = char_ids , target = u,  cond_mask=cond_mask, pred_mask=pred_mask)

        return loss

class CFM_Wrapper(BASECFM):
    def __init__(self, vocab_size):
        super().__init__()
        self.estimator = Temp(vocab_size=vocab_size+1)


