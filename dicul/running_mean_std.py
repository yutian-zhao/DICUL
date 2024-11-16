# Adapted from https://github.com/swan-utokyo/deir/blob/main/src/utils/running_mean_std.py by Wan ShanChuan
import torch as th


class RunningMeanStd(object):
    """
    Implemented based on:
    - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    - https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/mpi_util.py#L179-L214
    - https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
    """

    def __init__(self, device, epsilon=1e-4, momentum=None, shape=()):
        self.mean = th.zeros(size=shape, dtype=th.float64, device=device)
        self.var = th.ones(size=shape, dtype=th.float64, device=device)
        self.count = epsilon
        self.eps = epsilon
        self.momentum = momentum

    def clear(self):
        self.__init__(self.eps, self.momentum)

    @staticmethod
    def update_ema(old_data, new_data, momentum):
        if old_data is None:
            return new_data
        return old_data * momentum + new_data * (1.0 - momentum)

    def update(self, x):
        # batch_mean, batch_std, batch_count = th.mean(x, axis=0), th.std(x, axis=0), x.shape[0]
        batch_mean, batch_std, batch_count = th.mean(x), th.std(x), x.shape[0]
        batch_var = th.square(batch_std)
        if self.momentum is None or self.momentum < 0:
            self.update_from_moments(batch_mean, batch_var, batch_count)
        else:
            self.mean = self.update_ema(self.mean, batch_mean, self.momentum)
            new_var = th.mean(th.square(x - self.mean))
            self.var = self.update_ema(self.var, new_var, self.momentum)
            self.std = th.sqrt(self.var)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + th.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.std = th.sqrt(new_var)
        self.count = new_count
