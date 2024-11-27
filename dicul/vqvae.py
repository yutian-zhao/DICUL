"""
The implementation is adapted from the oringinal verision by Kaneko Tomoyuki
"""

from typing import Dict, Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from dicul.torch_util import FanInInitReLULayer


class VQCodebook(th.nn.Module):
    """https://pytorch.org/docs/stable/generated/th.nn.Embedding.html"""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.book = th.nn.Embedding(self.K, self.D)
        self.book.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def rdistances(self, x: th.Tensor) -> th.Tensor:
        """return relative distance to each code (y_i) for each x
        min_i (x-y_i) = min_i x^2 - 2x^T y_i + y_i^2 = min_i - 2x^T y_i + y_i^2
        """
        return -2 * x @ self.book.weight.T + th.sum(self.book.weight**2, dim=1)

    def indices(self, x: th.Tensor) -> th.Tensor:
        """return the index of the nearest codebook for each x"""
        # return (N)
        with th.no_grad():
            return th.min(self.rdistances(x), dim=-1).indices


class VQVAE(th.nn.Module):
    # Assume inputs of shape (N,D), and the ouput distribution is N(mean, I).
    def __init__(
        self,
        insize,
        outsize,
        hidsize,
        D=128,
        K=64,
        beta=0.25,
    ):
        super().__init__()
        self.encoder = FanInInitReLULayer(
            insize,
            D,
            layer_type="linear",
            init_scale=1.4,
        )
        # NOTE: inputs are codes and obs
        self.decoder = nn.Sequential(
            FanInInitReLULayer(
                D + insize,
                hidsize,
                layer_type="linear",
                init_scale=1.4,
            ),
            FanInInitReLULayer(
                hidsize,
                outsize,
                layer_type="linear",
                init_scale=1.4,
                use_activation=False,
            ),
        )
        self.book = VQCodebook(K, D)
        self.K = K
        self.D = D
        self.beta = beta

    def encode(self, x, just_indices=False):
        zs = self.encoder(x)
        shape = zs.shape
        indices = self.book.indices(zs)
        if just_indices:
            return indices
        codes = self.book.book(indices)
        embedding = th.nn.functional.mse_loss(zs.detach(), codes)
        commitment = th.nn.functional.mse_loss(zs, codes.detach())
        vq_loss = embedding + commitment * self.beta
        # straight-through estimator
        quantized_zs = zs + (codes - zs).detach()
        return quantized_zs, indices, vq_loss

    def forward(self, next_obs, obs):
        quantized_zs, indices, vq_loss = self.encode(next_obs)
        outputs = self.decoder(th.cat([obs, quantized_zs], axis=-1))
        return quantized_zs, outputs, vq_loss

    def decoder_loss(self, code_id, obs, targets, if_prob=False):
        # TODO: detach
        # NOTE: codes come after obs
        # (B,L,D), (B,L,D)
        obs_ndim = obs.ndim
        obs_shape = obs.shape
        outputs = self.decoder(th.cat([obs, self.book.book.weight.data[code_id].detach()], axis=-1))
        if if_prob:
            # nll = th.nn.functional.mse_loss(outputs, targets)  # +const
            gaussian = th.distributions.multivariate_normal.MultivariateNormal(
                outputs, th.eye(outputs.shape[-1], device=outputs.device)
            )
            logp = gaussian.log_prob(targets)  # used as reward # (B, L)

            # sum = None
            # for i in range(self.K):
            #     code = self.book.book.weight.data[i].detach()
            #     # (B,L,D)
            #     for _ in range(obs_ndim-1):
            #         code = code.unsqueeze(0)
            #     code = code.expand(*obs_shape[:-1], self.D)
            #     outputs_i = self.decoder(th.cat([obs, code], axis=-1))
            #     gaussian = th.distributions.multivariate_normal.MultivariateNormal(
            #         outputs_i, th.eye(outputs_i.shape[-1], device=outputs_i.device)
            #     )
            #     logp_i = gaussian.log_prob(targets)
            #     # intrinsic_reward = np.log(num_reps + 1) - np.log(1 + np.exp(np.clip(logp_altz - logp.reshape(1, -1), -50, 50)).sum(axis=0))
            #     if sum is not None:
            #         sum += th.exp(th.clip(logp_i-logp, -50, 50))
            #     else:
            #         sum = th.exp(th.clip(logp_i-logp, -50, 50))
            # intrinsic_reward = th.log(th.tensor(self.K)) - th.log(sum) # (B,N)

            # (B,L,K,D)
            obs = obs.unsqueeze(-2)
            targets = targets.unsqueeze(-2)
            code = self.book.book.weight.data.detach().unsqueeze(0).unsqueeze(0)
            logp = logp.unsqueeze(-1)
            target_shape = list(obs.shape)
            target_shape[-2] = self.K
            obs = obs.expand(*target_shape)
            targets = targets.expand(*target_shape)
            code = code.expand(*target_shape[:-1], self.D)
            logp = logp.expand(*target_shape[:-1])

            outputs_i = self.decoder(th.cat([obs, code], axis=-1))
            gaussian = th.distributions.multivariate_normal.MultivariateNormal(
                outputs_i, th.eye(outputs_i.shape[-1], device=outputs_i.device)
            )
            logp_i = gaussian.log_prob(targets) # (B,L,K)
            intrinsic_reward = th.log(th.tensor(self.K)) - th.log(th.exp(th.clip(logp_i - logp, -50, 50)).sum(axis=-1))

            # assert intrinsic_reward == intrinsic_reward_

            return intrinsic_reward
        else:
            nll = th.nn.functional.mse_loss(outputs, targets)  # +const
            # prob = None
            return nll

    def losspair(self, next_obs, obs, targets, if_prob=False, reduction="mean"):
        targets = targets.detach()
        # [N,D]
        quantized_zs, outputs, vq_loss = self(next_obs, obs)
        if if_prob:
            nll = th.nn.functional.mse_loss(outputs, targets, reduction=reduction)  # +const
            gaussian = th.distributions.multivariate_normal.MultivariateNormal(
                outputs, th.eye(outputs.shape[-1], device=outputs.device)
            )
            # TODO: DEBUG: use log prob
            prob = th.exp(gaussian.log_prob(targets))  # used as reward
        else:
            nll = th.nn.functional.mse_loss(outputs, targets, reduction=reduction)  # +const
            prob = None
        return nll, vq_loss, prob

    def loss(self, next_obs, obs, targets, if_prob=False, reduction="mean"):
        nll, vq_loss, prob = self.losspair(next_obs, obs, targets, if_prob=if_prob, reduction=reduction)
        return nll + vq_loss, prob
