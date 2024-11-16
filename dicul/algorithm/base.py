import abc
from typing import Dict

import torch as th

from dicul.model.base import BaseModel
from dicul.storage import RolloutStorage


class BaseAlgorithm(abc.ABC):
    def __init__(self, model: BaseModel):
        self.model = model

    @abc.abstractclassmethod
    def update(self, storage: RolloutStorage) -> Dict[str, th.Tensor]:
        pass
