import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from src.modules.transformer import Block
from src.modules.prompt import Prompt
from src.modules.utils import (
    FlattenHead,
    PoolingHead,
    RevIN,
)


@dataclass
class TEMPOConfig:
    """
    Configuration of a `TEMPO` model.

    Args:
        num_series: number of time series, N
        input_len: length of input time series, L
        pred_len: length of prediction time series, Y
        block_size: length of the block, which is the maximum length of the input.
            This is fixed by openai gpt2.
        n_layer: number of transformer layers
        n_head: number of heads in multihead attention
        n_embd: number of dimensions of embedding
        patch_size: size of the patch
        patch_stride: stride of the patch
        revin: whether to use RevIN
        affine: whether to use affine transformation in RevIN
        embd_pdrop: dropout rate for embedding layer
        resid_pdrop: dropout rate for residual connection
        attn_pdrop: dropout rate for attention layer
        head_type: type of the head, must be one of the keys in `head_types`
        head_pdtop: dropout rate for the head
        individual: whether to use individual head for each component
        lora: whether to use LoRA
        lora_config: configuration for LoRA
        model_type: type of the model, must be one of the keys in `params`
        interpret: whether to output components for interpretation
    """

    num_series: int
    input_len: int
    pred_len: int
    patch_size: int
    patch_stride: int
    block_size: int = None
    n_layer: int = None
    n_head: int = None
    n_embd: int = None
    revin: bool = True
    affine: bool = True
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    head_type: str = "flatten"
    head_pdtop: float = 0.1
    individual: bool = False
    lora: bool = False
    lora_config: dict = None
    prompt_config: dict = None
    model_type: str = "gpt2"
    interpret: bool = False

    def todict(self):
        return asdict(self)

    def __contains__(self, key):
        return key in self.todict()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def update(self, config: dict):
        for k, v in config.items():
            setattr(self, k, v)


class TEMPO(nn.Module):
    """
    Notation:
        B: batch size
        N: number of time series
        E: number of dimensions of embedding
        P: number of patches
        PS: size of the patch
        L: length of input time series
        Y: length of prediction time series
    """

    models = ("gpt2",)
    head_types = ("flatten", "pooling")

    params = {
        "gpt2": dict(block_size=1024, n_head=12, n_embd=768),
    }

    def __init__(self, config: TEMPOConfig):
        super().__init__()

        # parse config and save them as attributes
        for k, v in config.todict().items():
            setattr(self, k, v)

        self.revin = config.revin
        if self.revin:
            self.revin_layer = RevIN(
                self.num_series, affine=self.affine
            )  # here, num_series is 3 because we have 3 components

        self.patch_num = int((self.input_len - self.patch_size) / self.patch_stride + 1)
        self.patch_padding = False
        # padding to make sure the input length is divisible by patch_stride
        if (self.input_len - self.patch_size) % self.patch_stride != 0:
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.patch_stride))
            self.patch_num += 1
            self.patch_padding = True

        assert (
            self.patch_num <= self.block_size
        ), f"patch_num must be less than or equal to block_size: {self.block_size}"

        assert (
            self.model_type in self.models
        ), f"model_type must be one of {self.models}"

        self.promp_pool = Prompt(**self.prompt_config)  # prompt pool

        self.wte = nn.Linear(self.patch_size, self.n_embd)  # patch embedding

        self.enc_inp_len = (
            self.patch_num
            + self.prompt_config["top_k"] * self.prompt_config["prompt_length"]
        ) * self.num_series
        self.wpe = nn.Embedding(self.enc_inp_len, self.n_embd)

        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(self.embd_pdrop),
                h=nn.ModuleList(
                    [
                        Block(
                            n_embd=self.n_embd,
                            n_head=self.n_head,
                            attn_pdrop=self.attn_pdrop,
                            resid_pdrop=self.resid_pdrop,
                            block_size=self.block_size,
                            lora=self.lora,
                            lora_config=self.lora_config,
                        )
                        for _ in range(self.n_layer)
                    ]
                ),
                # ln_f=nn.LayerNorm(self.n_embd),
            )
        )

        self._init_head_cls()

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer)
                )

    def _init_head_cls(self):
        assert (
            self.head_type in self.head_types
        ), f"head_type must be one of {self.head_types}"

        if self.head_type == "flatten":
            self.head = FlattenHead(
                individual=self.individual,
                n_vars=self.num_series,
                nf=self.n_embd * self.patch_num,
                target_window=self.pred_len,
                head_dropout=self.head_pdtop,
            )
        elif self.head_type == "pooling":
            self.head = PoolingHead(
                n_embd=self.n_embd,
                target_window=self.pred_len,
                head_dropout=self.head_pdtop,
            )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, config: TEMPOConfig):
        """
        Initialize a pretrained LLM model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert "model_type" in config, "config must have a `model_type` field"
        if "n_layer" not in config:
            config["n_layer"] = 12  # openai gpt2 default
        model_type = config["model_type"]
        n_layer = config["n_layer"]
        config.update(cls.params[model_type])
        from transformers import GPT2LMHeadModel

        model = TEMPO(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # filer out unnecessary keys and layers deeper than n_layer
        drop_keys = ["wte", "wpe", "lm_head", "ln_f", "ln_1", "ln_2"] + [
            f"h.{i}." for i in range(n_layer, model_hf.config.n_layer)
        ]
        sd_hf = {k: v for k, v in sd_hf.items() if all(k_ not in k for k_ in drop_keys)}

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith("attn.masked_bias")]  # ignore these
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        # freeze the multihead attention layers and feedforward
        # layers by default.
        for n, p in model.named_parameters():
            if any(
                k in n for k in ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
            ):
                p.requires_grad = False

        return model

    def _pre_encoder(self, x: torch.Tensor):
        """
        Compute the input for transformer layers including normalization, patching, embedding, and prompt retrieval.

        Args:
            x: input data, shape (B, N, L)
        """
        # norm
        if self.revin:
            x = x.permute(0, 2, 1)  # B, L, N
            x = self.revin_layer(x, "norm")  # B, L, N
            x = x.permute(0, 2, 1)  # B, N, L

        # patching
        if self.patch_padding:
            x = self.padding_patch_layer(x)

        x = x.unfold(
            dimension=-1, size=self.patch_size, step=self.patch_stride
        )  # B, N, P, PS

        # embedding
        x_emb = self.wte(x)  # B, N, P, E

        # prompt retrieval
        x_prompt = self.promp_pool(x_emb)  # B, N, top_k, prompt_len, E
        B, N, top_k, prompt_len, E = x_prompt.shape
        x_prompt = x_prompt.reshape(B, N, top_k * prompt_len, E)
        # concatenate the prompt with the input
        x_emb = torch.cat([x_emb, x_prompt], dim=2)  # B, N, P + top_k * prompt_len, E

        return x_emb.view(B, -1, E)  # B, N * (P + top_k * prompt_len), E

    def _encoder(self, x_emb: torch.Tensor):
        """
        Compute the output of the transformer layers.

        Args:
            x: input data, shape (B, N * (P + top_k * prompt_len), E)
        """
        device = x_emb.device

        # position embedding
        pos = torch.arange(
            0, self.enc_inp_len, dtype=torch.long, device=device
        ).unsqueeze(
            0
        )  # 1, N * (P + top_k * prompt_len)
        pos_emb = self.wpe(pos)  # 1, N * (P + top_k * prompt_len), E

        # transformer layers
        h = self.transformer.drop(x_emb + pos_emb)  # B, N * (P + top_k * prompt_len), E

        for block in self.transformer.h:
            h = block(h)  # B, N * (P + top_k * prompt_len), E

        return h

    def _post_encoder(self, x_emb: torch.Tensor):
        """
        Compute the output of model including linear output layer, denormalization and concatenation.

        Args:
            x: input data, shape (B, N * (P + top_k * prompt_len), E)
        """
        B, _, E = x_emb.shape
        # extract each component embeddings
        x_emb = x_emb.view(B, self.num_series, -1, E)
        x_emb = x_emb[:, :, : self.patch_num, :]  # B, N, P, E

        # linear output layer
        out = self.head(x_emb)  # B, N, Y

        # denorm
        if self.revin:
            out = out.permute(0, 2, 1)  # B, Y, N
            out = self.revin_layer(out, "denorm")  # B, Y, N
            out = out.permute(0, 2, 1)  # B, N, Y

        if self.interpret:
            return out
        else:
            return out.sum(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor):
        """
        Compute the output and loss given the input data.

        Args:
            x: input data, shape (B, N, L)
        """
        # pre-encoder
        x = self._pre_encoder(x)  # B, N * (P + top_k * prompt_len), E

        # encoder
        h = self._encoder(x)  # B, N * (P + top_k * prompt_len), E

        # post-encoder
        out = self._post_encoder(h)

        return out

    @property
    def num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        n_params_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": n_params, "grad": n_params_grad}
