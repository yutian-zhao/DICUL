from typing import Dict, Optional, Sequence, Tuple
import math
from gym import spaces

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dicul.torch_util import FanInInitReLULayer


class DQNCNN(nn.Module):
    def __init__(
        self,
        # inchan: int,
        observation_space: spaces.Box,
        outsize: int,
        init_scale: float = 2,
        init_norm_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
    ):
        super().__init__()

        # Layers
        s = math.sqrt(init_scale)
        # TODO!: need to check
        inchan = observation_space.shape[0] # 3, 64, 64
        self.conv0 = FanInInitReLULayer(
            inchan,
            32,
            kernel_size=8,
            stride=4,
            padding=0,
            init_scale=s,
            **init_norm_kwargs,
        )
        self.conv1 = FanInInitReLULayer(
            32,
            64,
            kernel_size=4,
            stride=2,
            padding=0,
            init_scale=s,
            **init_norm_kwargs,
        )
        self.conv2 = FanInInitReLULayer(
            64,
            64,
            kernel_size=3,
            stride=1,
            padding=0,
            init_scale=s,
            **init_norm_kwargs,
        )

        self.cnn = nn.Sequential(self.conv0, self.conv1, self.conv2, nn.Flatten())

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.dense = FanInInitReLULayer(
            n_flatten,
            outsize,
            layer_type="linear",
            init_scale=s,
            **dense_init_norm_kwargs,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.dense(self.cnn(x))
