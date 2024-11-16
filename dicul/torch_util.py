from typing import Dict, Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FanInInitReLULayer(nn.Module):
    def __init__(
        self,
        inchan: int,
        outchan: int,
        layer_type: str = "conv",
        init_scale: float = 1.0,
        batch_norm: bool = False,
        batch_norm_kwargs: Dict = {},
        group_norm_groups: Optional[int] = None,
        layer_norm: bool = False,
        use_activation: bool = True,
        **layer_kwargs,
    ):
        super().__init__()

        # Normalization
        self.norm = None
        if batch_norm:
            self.norm = nn.BatchNorm2d(inchan, **batch_norm_kwargs)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, inchan)
        elif layer_norm:
            self.norm = nn.LayerNorm(inchan)

        # Layer
        layer = dict(conv=nn.Conv2d, conv3d=nn.Conv3d, linear=nn.Linear)[layer_type]
        self.layer = layer(inchan, outchan, bias=self.norm is None, **layer_kwargs)
        self.use_activation = use_activation

        # Initialization
        self.layer.weight.data *= init_scale / self.layer.weight.norm(
            dim=tuple(range(1, self.layer.weight.data.ndim)), p=2, keepdim=True
        )
        if self.layer.bias is not None:
            self.layer.bias.data *= 0

    def forward(self, x: th.Tensor):
        if self.norm is not None:
            x = self.norm(x)
        x = self.layer(x)
        if self.use_activation:
            x = F.relu(x, inplace=True)
        return x

class RNN(nn.Module):
    def __init__(
        self,
        inchan: int,
        outchan: int,
        init_scale: float = 1.0,
        batch_norm: bool = False,
        batch_norm_kwargs: Dict = {},
        group_norm_groups: Optional[int] = None,
        layer_norm: bool = False,
        use_activation: bool = False, # no activation
        dropout: float = 0.2,
        batch_first=True,
        **layer_kwargs,
    ):
        super().__init__()

        # Normalization
        self.norm = None
        if batch_norm:
            self.norm = nn.BatchNorm2d(inchan, **batch_norm_kwargs)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, inchan)
        elif layer_norm:
            self.norm = nn.LayerNorm(inchan)

        # Layer
        self.layer = nn.LSTM(inchan, outchan, bias=self.norm is None, dropout=dropout, batch_first=batch_first, **layer_kwargs) # batch_first=False, dropout=0.0, bidirectional=False, 
        self.use_activation = use_activation

        # Initialization
        for layer_weights in self.layer.all_weights:
            for weight in layer_weights:
                if weight.data.ndim > 1:
                    weight.data *= init_scale / weight.norm(
                        dim=tuple(range(1, weight.data.ndim)), p=2, keepdim=True
                    )
                elif self.layer.bias is not None:
                    weight.data *= 0

    def forward(self, x: th.Tensor, h: th.Tensor=None, c: th.Tensor=None):
        if self.norm is not None:
            x = self.norm(x)
        assert not (h is not None and c is None)
        assert not (h is None and c is not None)
        if h is not None and c is not None:
            x, (h, c) = self.layer(x, (h, c))
        else:
            x, (h, c) = self.layer(x)
        if self.use_activation:
            x = F.relu(x, inplace=True)
            h = F.relu(h, inplace=True)
            c = F.relu(classmethod, inplace=True)
        return x, (h, c)