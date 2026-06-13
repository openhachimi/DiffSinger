# BSD 3-Clause License
#
# Copyright (c) 2020, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import functools
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import betabinom


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias
        )
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal):
        return self.conv(signal)


class Invertible1x1ConvLUS(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        w, _ = torch.linalg.qr(torch.randn(c, c))
        if torch.det(w) < 0:
            w[:, 0] = -1 * w[:, 0]
        p, lower, upper = torch.lu_unpack(*torch.linalg.lu_factor(w))

        self.register_buffer('p', p)
        lower = torch.tril(lower, -1)
        lower_diag = torch.diag(torch.eye(c, c))
        self.register_buffer('lower_diag', lower_diag)
        self.lower = nn.Parameter(lower)
        self.upper_diag = nn.Parameter(torch.diag(upper))
        self.upper = nn.Parameter(torch.triu(upper, 1))

    def forward(self, z, reverse=False):
        u = torch.triu(self.upper, 1) + torch.diag(self.upper_diag)
        l = torch.tril(self.lower, -1) + torch.diag(self.lower_diag)
        w = torch.mm(self.p, torch.mm(l, u))
        if reverse:
            if not hasattr(self, 'W_inverse'):
                w_inverse = w.float().inverse()
                if z.type() == 'torch.cuda.HalfTensor':
                    w_inverse = w_inverse.half()
                self.W_inverse = w_inverse[..., None]
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        w = w[..., None]
        z = F.conv1d(z, w, bias=None, stride=1, padding=0)
        log_det_w = torch.sum(torch.log(torch.abs(self.upper_diag)))
        return z, log_det_w


class ConvAttention(torch.nn.Module):
    def __init__(self, n_mel_channels=80, n_speaker_dim=128,
                 n_text_channels=512, n_att_channels=80, temperature=1.0,
                 n_mel_convs=2, align_query_enc_type='3xconv',
                 use_query_proj=True):
        super().__init__()
        self.temperature = temperature
        self.att_scaling_factor = np.sqrt(n_att_channels)
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.align_query_enc_type = align_query_enc_type
        self.use_query_proj = bool(use_query_proj)

        self.key_proj = nn.Sequential(
            ConvNorm(n_text_channels, n_text_channels * 2, kernel_size=3, bias=True, w_init_gain='relu'),
            torch.nn.SiLU(),
            ConvNorm(n_text_channels * 2, n_att_channels, kernel_size=1, bias=True)
        )

        if align_query_enc_type == "inv_conv":
            self.query_proj = Invertible1x1ConvLUS(n_mel_channels)
        elif align_query_enc_type == "3xconv":
            self.query_proj = nn.Sequential(
                ConvNorm(n_mel_channels, n_mel_channels * 2, kernel_size=3, bias=True, w_init_gain='relu'),
                torch.nn.SiLU(),
                ConvNorm(n_mel_channels * 2, n_mel_channels, kernel_size=1, bias=True),
                torch.nn.SiLU(),
                ConvNorm(n_mel_channels, n_att_channels, kernel_size=1, bias=True)
            )
        else:
            raise ValueError("Unknown query encoder type specified")

    def forward(self, queries, keys, query_lens=None, mask=None, key_lens=None,
                keys_encoded=None, attn_prior=None):
        keys_enc = self.key_proj(keys)

        if self.use_query_proj:
            if self.align_query_enc_type == "inv_conv":
                queries_enc, log_det_w = self.query_proj(queries)
            elif self.align_query_enc_type == "3xconv":
                queries_enc = self.query_proj(queries)
                log_det_w = 0.0
            else:
                queries_enc, log_det_w = self.query_proj(queries)
        else:
            queries_enc, log_det_w = queries, 0.0

        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2
        attn = -0.0005 * attn.sum(1, keepdim=True)

        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-4)

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2), -1e+4)

        attn = self.softmax(attn + 1e-4)
        return attn, attn_logprob


def mas_width1(log_attn_map):
    t, p = log_attn_map.shape
    log_p = log_attn_map.copy()
    log_p[0, 1:] = -np.inf

    for i in range(1, t):
        prev = log_p[i - 1]
        prev_shifted = np.full(p, -np.inf)
        prev_shifted[1:] = prev[:-1]
        log_p[i] += np.maximum(prev_shifted, prev)

    opt = np.zeros_like(log_p)
    j = p - 1
    for i in range(t - 1, 0, -1):
        opt[i, j] = 1
        if j > 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
            j -= 1
            if j == 0:
                opt[1:i, j] = 1
                break
    opt[0, j] = 1
    return opt


def binarize_attention(attn, in_lens, out_lens):
    b_size = attn.shape[0]
    with torch.no_grad():
        attn_out_cpu = np.zeros(attn.data.shape, dtype=np.float32)
        log_attn_cpu = torch.log(attn.data).to(device='cpu', dtype=torch.float32).numpy()
        out_lens_cpu = out_lens.cpu()
        in_lens_cpu = in_lens.cpu()
        for ind in range(b_size):
            hard_attn = mas_width1(
                log_attn_cpu[ind, 0, :out_lens_cpu[ind], :in_lens_cpu[ind]]
            )
            attn_out_cpu[ind, 0, :out_lens_cpu[ind], :in_lens_cpu[ind]] = hard_attn
        attn_out = torch.tensor(attn_out_cpu, device=attn.device, dtype=attn.dtype)
    return attn_out


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling=1.0):
    p = phoneme_count
    m = mel_count
    x = np.arange(0, p)
    mel_text_probs = []
    for i in range(1, m + 1):
        a, b = scaling * i, scaling * (m + 1 - i)
        rv = betabinom(p, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


class BetaBinomialInterpolator:
    def __init__(self, round_mel_len_to=100, round_text_len_to=20):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        self.bank = functools.lru_cache(beta_binomial_prior_distribution)

    def round(self, val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = self.round(w, to=self.round_mel_len_to)
        bh = self.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret


class AttentionCTCLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.blank_logprob = blank_logprob
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        max_key_len = attn_logprob.size(-1)

        attn_logprob = attn_logprob.squeeze(1).permute(1, 0, 2)
        attn_logprob = F.pad(
            input=attn_logprob,
            pad=(1, 0, 0, 0, 0, 0),
            value=self.blank_logprob
        )
        key_inds = torch.arange(max_key_len + 1, device=attn_logprob.device, dtype=torch.long)
        attn_logprob.masked_fill_(
            key_inds.view(1, 1, -1) > key_lens.view(1, -1, 1),
            -1e+4
        )
        attn_logprob = self.log_softmax(attn_logprob)
        target_seqs = key_inds[1:].unsqueeze(0).repeat(key_lens.numel(), 1)
        cost = self.ctc_loss(
            attn_logprob, target_seqs,
            input_lengths=query_lens, target_lengths=key_lens
        )
        return cost


class AttentionBinarizationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention, eps=1e-4):
        log_sum = torch.log(
            torch.clamp(soft_attention[hard_attention == 1], min=eps)
        ).sum()
        return -log_sum / hard_attention.sum()


class AlignmentLearningFramework(torch.nn.Module):
    def __init__(self, feature_size: int, encoding_size: int):
        super().__init__()
        self.attention = ConvAttention(
            feature_size,
            0,
            encoding_size,
            use_query_proj=True,
            align_query_enc_type='3xconv'
        )

    def forward(
            self,
            token_embeddings: torch.Tensor,
            encoding_lengths: torch.Tensor,
            features: torch.Tensor,
            feature_lengths: torch.Tensor,
            attention_priors: torch.Tensor
    ):
        attention_masks = mask_from_lens(encoding_lengths, max_len=encoding_lengths.max())
        attention_masks = attention_masks[..., None] == 0

        attention_softs, attention_logprobs = self.attention(
            queries=features,
            keys=token_embeddings,
            mask=attention_masks,
            attn_prior=attention_priors
        )
        attention_hards = binarize_attention(attention_softs, encoding_lengths, feature_lengths)
        durations = attention_hards.sum(2)[:, 0, :]
        assert torch.all(torch.eq(durations.sum(dim=1), feature_lengths))
        return durations, attention_softs, attention_hards, attention_logprobs
