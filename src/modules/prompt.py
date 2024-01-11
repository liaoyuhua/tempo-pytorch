from typing import Union
import torch
import torch.nn as nn


def get_prompt_param_cls(
    shape: tuple, prompt_init: str = "uniform", share: bool = True
):
    """
    Args:
        shape: shape of prompt parameter
        prompt_init: one of "zero", "uniform"
        share: whether to share prompt across components, default True
    """
    if prompt_init == "zero":
        if share:
            return nn.Parameter(torch.zeros(shape))
        else:
            return nn.ParameterList([nn.Parameter(torch.zeros(shape))] * 3)
    elif prompt_init == "uniform":
        if share:
            return nn.init.uniform_(nn.Parameter(torch.randn(shape)), -1, 1)
        else:
            return nn.ParameterList(
                [nn.init.uniform_(nn.Parameter(torch.randn(shape)), -1, 1)] * 3
            )
    else:
        raise ValueError(f"Unknown prompt initialization type: {prompt_init}")


def get_prompt_value(
    prompt: Union[nn.Parameter, nn.ParameterList], idx: torch.Tensor, share: bool = True
) -> torch.Tensor:
    """
    Args:
        prompt: prompt parameter of shape (pool_size, prompt_length, embed_dim) or
            list of prompt parameters of shape (pool_size, prompt_length, embed_dim)
        idx: index tensor of shape (batch_size, num_components, top_k)
        share: whether to share prompt across components, default True

    Returns:
        torch.Tensor: prompt value of shape (batch_size, num_components, top_k, prompt_length, embed_dim)
    """
    if share:
        return prompt[idx]  # B, N, top_k, P, E
    else:
        return torch.stack(
            [prompt[idx[:, idx_]] for idx_ in range(idx.shape[1])],  # B, top_k, P, E
            dim=1,
        )  # B, N, top_k, P, E


class Prompt(nn.Module):
    def __init__(
        self,
        prompt_length: int = 3,
        pool_size: int = 30,
        embed_dim: int = None,
        embedding_key="mean",
        top_k: int = 3,
        key_init: str = "uniform",
        prompt_init: str = "uniform",
        prompt_share: bool = True,
    ):
        """
        Args:
            prompt_length: length of prompt
            pool_size: size of prompt pool
            embed_dim: embedding dimension
            embedding_key: one of "mean", "max", "mean_max"
            top_k: top k prompts to use
            key_init: one of "zero", "uniform"
            prompt_init: one of "zero", "uniform"
            promp_share: whether to share prompt across components, default True
        """
        super().__init__()
        self.prompt_length = prompt_length
        self.pool_size = pool_size
        self.embedding_key = embedding_key

        self.top_k = top_k

        self.prompt_key_shape = (pool_size, embed_dim)
        self.prompt_value_shape = (pool_size, prompt_length, embed_dim)
        self.prompt_share = prompt_share

        self._init_weights(key_init, prompt_init, prompt_share)

    def _init_weights(
        self, key_init: str = None, prompt_init: str = None, prompt_share: bool = None
    ):
        self.prompt_key = get_prompt_param_cls(
            self.prompt_key_shape, key_init, prompt_share
        )
        self.prompt_value = get_prompt_param_cls(
            self.prompt_value_shape, prompt_init, prompt_share
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: B, N, P, E

        Returns:
            torch.Tensor: prompt value of shape (batch_size, num_components, top_k, prompt_length, embed_dim)
        """
        # B, N, E
        if self.embedding_key == "mean":
            x = torch.mean(x, dim=2)
        elif self.embedding_key == "max":
            x = torch.max(x, dim=2)[0]
        elif self.embedding_key == "mean_max":
            x = torch.max(x, dim=2)[0] + 2 * torch.mean(x, dim=2)
        else:
            raise NotImplementedError(
                "Not supported way of calculating embedding keys!"
            )

        if self.prompt_share:
            similarity = torch.matmul(x, self.prompt_key.T)  # B, N, Pool_size
        else:
            similarity = torch.stack(
                [
                    torch.matmul(x[:, idx, :], key.T)  # B, 1, Pool_size
                    for idx, key in enumerate(self.prompt_key)
                ],
                dim=1,
            )  # B, N, Pool_size

        _, idx = torch.topk(similarity, self.top_k, dim=-1)  # B, N, top_k

        prompt = get_prompt_value(self.prompt_value, idx, self.prompt_share)

        return prompt
