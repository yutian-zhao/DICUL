from collections import namedtuple, deque
import copy
import random
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch as th
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# from dicul.algorithm.base import BaseAlgorithm
# from dicul.model.ppo_ad import PPOADModel
# from dicul.storage import RolloutStorage

PriorityTraj = namedtuple("PriorityTraj", ("priority", "r_e", "rec_error", "reach_prob", "traj"))


class PrioritizedBuffer:
    """
    PrioritizedBuffer
    """

    def __init__(self, maxlen: int = 1000):
        # TODO: can be more efficient by implementing based on tensors and pointers
        self.count_ids: List[int] = deque(maxlen=maxlen)
        self.priorities: List[float] = deque(maxlen=maxlen)
        self.visit_counts: List[int] = deque(maxlen=maxlen)
        self.r_es: List[th.Tensor] = deque(maxlen=maxlen)
        self.rec_errors: List[th.Tensor] = deque(maxlen=maxlen)
        self.reach_probs: List[th.Tensor] = deque(maxlen=maxlen)
        self.trajs: List[th.Tensor] = deque(maxlen=maxlen)
        self.count = 0

    def __len__(self):
        return len(self.trajs)

    def min_id(self):
        if self.count == 0:
            return 0
        return self.count_ids[0]

    def append(self, ptraj: PriorityTraj):
        priority, r_e, rec_error, reach_prob, traj = ptraj
        self.count_ids.append(self.count)
        self.priorities.append(priority)
        self.visit_counts.append(0)
        self.r_es.append(r_e)
        self.rec_errors.append(rec_error)
        self.reach_probs.append(reach_prob)
        self.trajs.append(traj)
        self.count += 1

    def update(self, ind, **kwargs):
        # NOTE: indices are only valid before next append
        for key, value in kwargs.items():
            getattr(self, key)[ind] = value
        self.update_priority(ind)

    def update_priority(self, ind):
        self.priorities[ind] = (
            10 * (0.8 ** self.visit_counts[ind]) + (1 - self.rec_errors[ind]) + self.r_es[ind]
        )  # 0.9*(1-self.reach_probs[ind])

    def sample(self, n):
        # NOTE: sampled indices are only valid before next append
        # sampled_indices = th.multinomial(self.priorities, n, replacement=True)
        # self.visit_counts[sampled_indices] += 1
        # TODO!: priority need to caculate on using or on updates.
        indices = random.choices(range(len(self)), k=n, weights=self.priorities)
        ids = []
        for ind in indices:
            self.visit_counts[ind] += 1
            ids.append(self.count_ids[ind])  # min_id+ids
            self.update_priority(ind)
        # TODO!: return indices and id
        return indices, ids


if __name__ == "__main__":
    pass
    # pbuffer = PrioritizedBuffer(maxlen=3)
    # for i in range(4):
    #     ptraj = PriorityTraj(i,i,i,i,i,)
    #     pbuffer.append(ptraj)
    # print(pbuffer.count_ids)
    # print(pbuffer.sample(5))

    import torch as th
    import torch.optim as optim

    # a=th.ones((3,2))
    # a[1] = 0
    # b= th.tensor([0,1,0,0,1])
    # print(a[b], a[b].shape)
    # a = th.ones((5, 1))
    # b = th.tensor([0, 1, 0, 0, 1]).bool()
    # a[b] = th.zeros((2, 1))
    # print(a)
    # a = th.randn(3, 2)
    # print(a)
    # print(th.min(a, dim=1).indices)
    # from torch.nn.utils.rnn import pad_sequence

    # a = th.tensor([1, 2, 3])
    # b = th.tensor([0, 1])
    # c = th.tensor([0, 5, 6, 8])

    # # B,T,1
    # # pytorch v2.5 ,padding_side='right'
    # padded = pad_sequence([a, b, c], batch_first=True, padding_value=0.0)
    # print(padded)

    # masks_p=th.tensor([0,1,0,1,0])
    # done_conds_p = (masks_p == 0).squeeze(dim=-1)
    # done_steps_p = done_conds_p.nonzero().squeeze(dim=-1)
    # done_steps_p = done_steps_p.tolist()
    # print(done_steps_p)
    # done_steps_p = sorted(done_steps_p)
    # print(done_steps_p)

    a = th.ones((3, 2))
    a[1] = 0
    b = th.tensor([[1, 0, 0], [0, 0, 1]])
    print(a[b])
    exit()
    rnn = th.nn.LSTM(10, 20, 2)
    encoder = th.nn.Linear(10, 10)
    linear1 = th.nn.Linear(20, 20)
    linear2 = th.nn.Linear(20, 20)
    optimizer = optim.Adam(rnn.parameters())
    input = th.randn(5, 3, 10)
    input = encoder(input)
    output, (hn, cn) = rnn(input)
    output, _ = rnn(input, (hn, cn))
    loss_function = th.nn.GaussianNLLLoss()
    loss1 = loss_function(linear1(output), th.ones_like(output), th.eye(output))
    print(loss1)
    output, (hn, cn) = rnn(input, (th.ones_like(hn), th.ones_like(cn)))
    output, _ = rnn(input, (hn, cn))
    loss2 = th.nn.functional.mse_loss(linear2(output), th.ones_like(output))
    loss = loss1 + loss2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(prev_hn is hn)
    # output, (hn, cn) = rnn(input,(prev_hn, prev_cn))
    # loss = th.nn.functional.mse_loss(output, th.ones_like(output))
    # loss.backward()
    # optimizer.step()
    # output, (hn, cn) = rnn(input,(hn, cn))
    print(output)
