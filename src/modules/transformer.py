from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import Transpose
from .lora import MergedLinear


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class CausalSelfAttention(nn.Module):
    """
    Multi-headed attention layer with causal mask. This module can be implemented with or without LoRA.
    """

    def __init__(
        self,
        block_size: int,
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
        lora: bool = True,
        lora_r: int = None,
        lora_alpha: int = None,
        lora_dropout: float = None,
        enable_lora: List[bool] = None,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
    ):
        """
        Args:
            block_size: length of the block, which is the maximum length of the input.
                This is fixed by openai gpt2.
            n_embd: number of embedding dimensions
            n_head: number of attention heads
            attn_pdrop: dropout rate for attention layer
            resid_pdrop: dropout rate for residual connection
            lora: whether to use LoRA, default True
            lora_r: rank of the LoRA approximation
            lora_alpha: scaling factor for LoRA
            lora_dropout: dropout rate for LoRA
            enable_lora: whether to enable LoRA for q, k, v, default [True, True, False]
            fan_in_fan_out: whether to transpose the weight matrix, default False
            merge_weights: whether to merge the LoRA weights into the weight matrix, default False
        """
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        if lora:
            enable_lora = (
                [True, True, False] if enable_lora is None else enable_lora
            )  # default: enable lora for q, k not v
            self.c_attn = MergedLinear(
                n_embd,
                n_embd * 3,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                enable_lora=enable_lora,
                fan_in_fan_out=fan_in_fan_out,
                merge_weights=merge_weights,
            )
        else:
            self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    norm_types = ["batch", "layer"]

    def __init__(
        self,
        block_size: int,
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
        norm: str = "batch",
        lora: bool = True,
        lora_config: dict = None,
    ):
        """
        Basic block of the transformer. This module can be implemented with or without LoRA.

        Args:
            block_size: length of the block, which is the maximum length of the input.
                This is fixed by openai gpt2.
            n_embd: number of embedding dimensions
            n_head: number of attention heads
            attn_pdrop: dropout rate for attention layer
            resid_pdrop: dropout rate for residual connection
            norm: type of normalization, either "batch" or "layer"
            lora: whether to use LoRA, default True
            lora_config: configuration for LoRA, default None
        """
        super().__init__()
        assert norm in self.norm_types, f"norm type must be one of {self.norm_types}"
        if norm == "batch":
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(n_embd), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(n_embd)

        lora_config = {} if lora_config is None else lora_config
        self.attn = CausalSelfAttention(
            block_size=block_size,
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            lora=lora,
            **lora_config,
        )

        if norm == "batch":
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(n_embd), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(n_embd)

        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlpf(self.norm_ffn(x))
        return x
