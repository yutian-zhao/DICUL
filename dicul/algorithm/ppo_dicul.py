import copy
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from gym import spaces
from collections import defaultdict

from dicul.model.ppo_dicul import PPODICULModel
from dicul.algorithm.base import BaseAlgorithm
from dicul.storage import RolloutStorage
from dicul.running_mean_std import RunningMeanStd
from dicul.prioritized_buffer import PriorityTraj, PrioritizedBuffer


class TrajBuffer:
    """
    Traj: obs, skills, actions, log_probs, skill_log_probs, advs, skill_advs, returns, skill_returns,
    """

    # chunk
    def __init__(
        self,
        nstep: int,
        nproc: int,
        # model: PPODICULModel,
        # nbatch: int,
        # observation_space: spaces.Box,
        action_space: spaces.Discrete,
        device: th.device,
        max_skill_len: int,
        min_skill_len: int,
        pbuffer: PrioritizedBuffer,
    ):
        # Params
        # self.nbatch = nbatch
        self.nstep = nstep
        self.nproc = nproc
        # self.model = model
        self.device = device
        self.num_actions = getattr(action_space, "n")
        # NOTE: check in-place modification
        self.pbuffer = pbuffer
        self.min_skill_len = min_skill_len
        self.max_skill_len = max_skill_len

        # Get obs shape and action dim
        # assert isinstance(observation_space, spaces.Box)
        # assert isinstance(action_space, spaces.Discrete), type(action_space)
        # obs_shape = getattr(observation_space, "shape")
        # action_shape = (1,)

        self.master_policy_data = defaultdict(list)
        self.skill_policy_data = defaultdict(list)
        self.aux_skill_policy_data = []
        # save padded traj data
        self.master_policy_padded_data = {}
        self.skill_policy_padded_data = {}
        # save incomplete trajs
        self.master_policy_buffer = defaultdict(lambda: [None for _ in range(self.nproc)])
        self.skill_policy_buffer = defaultdict(lambda: [None for _ in range(self.nproc)])
        self.master_policy_entries = [
            "obs",
            "skills",
            "log_probs",
            "advs",
            "returns",
            "masks",
        ]
        self.skill_policy_entries = [
            "obs",
            "skills",
            "actions",
            "skill_log_probs",
            "advs",
            "skill_advs",
            "skill_returns",
            "masks",
            "traj_ids",
            "master_intrinsic_rewards",
            "rewards",
        ]

        self.mean_rec_error = RunningMeanStd(device=device, shape=(1,))

    def process_data(self, storage: RolloutStorage):
        # TODO!: Check min skill length
        # TODO!: extract skill trajectories with high priority to priority buffer
        # TODO: Compare GPU and CPU
        # TODO: need to check correctness
        # TODO: contiguous
        # update mean_rec_error
        self.mean_rec_error.update(storage.master_intrinsic_rewards)

        # Get done steps
        self.aux_skill_policy_new_data_start = len(self.aux_skill_policy_data)
        for p in range(self.nproc):
            # process master trajectories
            masks_p = storage.masks[: self.nstep, p]
            done_conds_p = (masks_p == 0).squeeze(dim=-1)
            done_steps_p = done_conds_p.nonzero().squeeze(dim=-1)
            # If indices_or_sections is a tensor, it must be a zero-dimensional or one-dimensional long tensor on the CPU.
            done_steps_p = th.sort(done_steps_p)[0].long().to("cpu")

            for m_entry in self.master_policy_entries:
                data_p = getattr(storage, m_entry)[: self.nstep, p]
                data_p_split = th.tensor_split(data_p, done_steps_p, dim=0)  # tuple of tensors (steps, D)
                data_p_split = list(data_p_split)
                if self.master_policy_buffer[m_entry][p] is not None:
                    data_p_split[0] = th.cat([self.master_policy_buffer[m_entry][p], data_p_split[0]], dim=0)
                # if done_steps_p[-1] < self.nstep - 1:
                self.master_policy_buffer[m_entry][p] = data_p_split.pop(-1)
                # else:
                #     self.master_policy_buffer[m_entry][p] = None
                if len(data_p_split[0]) == 0:
                    data_p_split.pop(0)
                self.master_policy_data[m_entry] += data_p_split

            # process skill trajectories
            skill_step_counts_p = storage.skill_step_counts[: self.nstep, p]
            skill_step_counts_p = (skill_step_counts_p == 1).squeeze(dim=-1)
            skill_step_counts_p = skill_step_counts_p.nonzero().squeeze(dim=-1)
            skill_step_counts_p = th.sort(skill_step_counts_p)[0].long().to("cpu")

            for s_entry in self.skill_policy_entries:
                # NOTE: pointer
                data_p = getattr(storage, s_entry)[: self.nstep, p]
                data_p_split = th.tensor_split(
                    data_p, skill_step_counts_p, dim=0
                )  # list of tensors (steps, D)
                data_p_split = list(data_p_split)
                if self.skill_policy_buffer[s_entry][p] is not None:
                    data_p_split[0] = th.cat([self.skill_policy_buffer[s_entry][p], data_p_split[0]], dim=0)
                
                if len(data_p_split[0]) == 0:
                    data_p_split.pop(0)

                if s_entry == "obs":
                    # prepare aux data
                    # mark starting point for traj storation
                    for i in range(len(data_p_split) - 1):
                        skill_traj = th.cat([data_p_split[i], data_p_split[i + 1][0].unsqueeze(0)], dim=0)
                        # NOTE: Ensure no in-place change
                        assert len(skill_traj) <= self.max_skill_len+1
                        self.aux_skill_policy_data.append(skill_traj)

                self.skill_policy_buffer[s_entry][p] = data_p_split.pop(-1)
                self.skill_policy_data[s_entry] += data_p_split

        self.aux_skill_policy_new_data_end = len(self.aux_skill_policy_data)


        # prepare aux data
        # skill traj + choosen skill
        # add selected skill traj
        # NOTE: must done before new trajs storation
        min_id = self.pbuffer.min_id()
        for traj_id in self.skill_policy_data["traj_ids"]:
            traj_id = traj_id[0].squeeze()
            if traj_id >= 0 and traj_id >= min_id:
                self.aux_skill_policy_data.append(self.pbuffer.trajs[traj_id - min_id])

        # add the beginning of the next skill to the end of the current skill
        # TODO: check id and indices match for traj_id
        # TODO: ensure indices not change, i.e., no insertion during processing and updating
        # for i, traj in enumerate(skill_step_counts_p):
        assert (
            self.aux_skill_policy_new_data_start + len(self.skill_policy_data["obs"])
            == self.aux_skill_policy_new_data_end
        )
        for i in range(len(self.skill_policy_data["obs"])):
            skill_traj = self.aux_skill_policy_data[self.aux_skill_policy_new_data_start + i]

            # Save novel trajs
            len_traj = len(skill_traj)
            max_extrinsic_reward = th.max(self.skill_policy_data["rewards"][i])
            reach_probability = th.prod(self.skill_policy_data["skill_log_probs"][i])
            random_reach_probability = (1 / self.num_actions) ** len_traj
            max_rec_error = th.max(self.skill_policy_data["master_intrinsic_rewards"][i])
            if len_traj - 1 >= self.min_skill_len and (
                max_extrinsic_reward > 0.1 or (max_rec_error > self.mean_rec_error.mean)
            ):
                # reach_probability<random_reach_probability
                # TODO! record
                priority = (
                    max_extrinsic_reward + 10 * 0.7 + (1 - max_rec_error)
                )  # +0.9*(1-random_reach_probability)
                ptraj = PriorityTraj(
                    priority, max_extrinsic_reward, max_rec_error, reach_probability, skill_traj
                )
                assert len_traj <= self.max_skill_len+1
                self.pbuffer.append(ptraj)

        # pad
        # TODO: assert len are the same
        # TODO: use done_steps to be more efficient
        # TODO: check padding 0
        # TODO: use a non-zero tensor to get masks
        eps_max_len, lens = self.get_len(self.master_policy_data["obs"])
        for key, value in self.master_policy_data.items():
            # Except trajectories, the need to pad?
            if key == "masks":
                self.master_policy_padded_data[key] = (
                    nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=-1).contiguous() > -1
                ).float()
            else:
                self.master_policy_padded_data[key] = nn.utils.rnn.pad_sequence(
                    value, batch_first=True, padding_value=0.0
                ).contiguous()  # B x T x * Tensor .squeeze()

        # pad skill trajs
        # TODO: unify master and skill otherwise easy to make mistakes
        skill_max_len, lens = self.get_len(self.skill_policy_data["obs"])
        for key, value in self.skill_policy_data.items():
            # Except trajectories, the need to pad?
            if key == "masks":
                self.skill_policy_padded_data[key] = (
                    nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=-1).contiguous() > -1
                ).float()
            else:
                self.skill_policy_padded_data[key] = nn.utils.rnn.pad_sequence(
                    value, batch_first=True, padding_value=0.0
                ).contiguous()  # B x T x * Tensor .squeeze()?

    def get_len(self, list_of_seqs):
        lens = []  # must be on cpu
        max_len = -1
        for i in list_of_seqs:
            len_i = len(i)
            lens.append(len_i)
            if len_i > max_len:
                max_len = len_i
        return max_len, lens

    def master_policy_data_loader(self, nbatch):
        # TODO!: Temperally treat nbatch as batchsize
        ndata = len(self.master_policy_padded_data["obs"])
        # assert ndata >= nbatch
        # batch_size = ndata // nbatch
        sampler = SubsetRandomSampler(range(ndata))
        sampler = BatchSampler(sampler, batch_size=2, drop_last=False)
        return sampler

    def skill_policy_data_loader(self, nbatch):
        # TODO: seperate skill batch (meaning?)
        ndata = len(self.skill_policy_padded_data["obs"])
        # assert ndata >= nbatch
        # batch_size = ndata // nbatch
        sampler = SubsetRandomSampler(range(ndata))
        sampler = BatchSampler(sampler, batch_size=nbatch, drop_last=False)
        return sampler

    def aux_skill_policy_data_loader(self, nbatch):
        ndata = len(self.aux_skill_policy_data)
        # assert ndata >= nbatch
        # batch_size = ndata // nbatch
        sampler = SubsetRandomSampler(range(ndata))
        sampler = BatchSampler(sampler, batch_size=nbatch, drop_last=False)
        return sampler

    def reset_policy_data(
        self,
    ):
        # TODO: reset data not buffer
        self.master_policy_data.clear()
        self.skill_policy_data.clear()
        self.master_policy_padded_data.clear()
        self.skill_policy_padded_data.clear()

    def reset_aux_data(
        self,
    ):
        self.aux_skill_policy_data = []


class PPODICULAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        model: PPODICULModel,
        ppo_nepoch: int,
        ppo_nbatch: int,
        clip_param: float,
        vf_loss_coef: float,
        ent_coef: float,
        lr: float,
        max_grad_norm: float,
        aux_freq: int,
        aux_nepoch: int,
        aux_nbatch: int,
        pi_dist_coef: int,
        vf_dist_coef: int,
        max_skill_len: int,
        min_skill_len: int,
        nstep: int,
        nproc: int,
        device: th.device,
    ):
        super().__init__(model)
        self.model = model

        # PPO params
        self.clip_param = clip_param
        self.ppo_nepoch = ppo_nepoch
        self.ppo_nbatch = ppo_nbatch
        self.vf_loss_coef = vf_loss_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_count = 0
        self.max_skill_len = max_skill_len
        self.min_skill_len = min_skill_len
        self.nstep = nstep
        self.nproc = nproc
        self.device = device

        # Aux params
        self.aux_freq = aux_freq
        self.aux_nepoch = aux_nepoch
        self.aux_nbatch = aux_nbatch
        self.pi_dist_coef = pi_dist_coef
        self.vf_dist_coef = vf_dist_coef

        self.buffer = TrajBuffer(
            nstep=self.nstep,
            nproc=self.nproc,
            action_space=self.model.action_space,
            min_skill_len=self.min_skill_len,
            max_skill_len=self.max_skill_len,
            pbuffer=self.model.pbuffer,
            device=self.device,
        )

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        # NOTE: necessary? should seperate vqvae
        self.aux_optimizer = optim.Adam(model.parameters(), lr=lr)

    def update(self, storage: RolloutStorage):
        # Set model to training mode
        self.model.train()
        self.buffer.process_data(storage)

        # Run PPO
        pi_loss_epoch = 0
        vf_loss_epoch = 0
        entropy_epoch = 0
        nupdate = 0
        skill_pi_loss_epoch = 0
        skill_vf_loss_epoch = 0
        skill_entropy_epoch = 0
        skill_nupdate = 0

        for _ in range(self.ppo_nepoch):
            # Get data loader
            # 512*8 steps, 512 bs * 8 nb, 8nb*3epoch updates,
            # 200*20traj steps, 4nb*(5*200)bs, 4*6epoch updates, overfitting?
            # 25*160traj steps, 8nb detach?
            # TODO: the master policy should be updated later and faster than skill policies? Currently update simultaneously
            data_loader = self.buffer.skill_policy_data_loader(self.ppo_nbatch)
            for indices in data_loader:
                # Compute loss
                skill_losses = self.model.compute_losses(
                    mode="skill",
                    obs=self.buffer.skill_policy_padded_data["obs"][indices],
                    advs=self.buffer.skill_policy_padded_data["skill_advs"][indices],
                    actions=self.buffer.skill_policy_padded_data["actions"][indices],
                    skills=self.buffer.skill_policy_padded_data["skills"][indices],
                    log_probs=self.buffer.skill_policy_padded_data["skill_log_probs"][indices],
                    vtargs=self.buffer.skill_policy_padded_data["skill_returns"][indices],
                    masks=self.buffer.skill_policy_padded_data["masks"][indices],
                    clip_param=self.clip_param,
                )
                skill_pi_loss = skill_losses["pi_loss"]
                skill_vf_loss = skill_losses["vf_loss"]
                skill_entropy = skill_losses["entropy"]
                skill_loss = skill_pi_loss + self.vf_loss_coef * skill_vf_loss - self.ent_coef * skill_entropy

                # Update stats
                skill_pi_loss_epoch += skill_pi_loss.item()
                skill_vf_loss_epoch += skill_vf_loss.item()
                skill_entropy_epoch += skill_entropy.item()
                skill_nupdate += 1

                self.optimizer.zero_grad()
                skill_loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            data_loader = self.buffer.master_policy_data_loader(self.ppo_nbatch)
            for indices in data_loader:
                # Compute loss
                losses = self.model.compute_losses(
                    mode="master",
                    obs=self.buffer.master_policy_padded_data["obs"][indices],
                    advs=self.buffer.master_policy_padded_data["advs"][indices],
                    actions=self.buffer.master_policy_padded_data["skills"][indices],
                    log_probs=self.buffer.master_policy_padded_data["log_probs"][indices],
                    vtargs=self.buffer.master_policy_padded_data["returns"][indices],
                    masks=self.buffer.master_policy_padded_data["masks"][indices],
                    clip_param=self.clip_param,
                )
                pi_loss = losses["pi_loss"]
                vf_loss = losses["vf_loss"]
                entropy = losses["entropy"]
                loss = pi_loss + self.vf_loss_coef * vf_loss - self.ent_coef * entropy
                # loss += skill_loss

                # Update parameter
                self.optimizer.zero_grad()
                # TODO!: backward a second time
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update stats
                pi_loss_epoch += pi_loss.item()
                vf_loss_epoch += vf_loss.item()
                entropy_epoch += entropy.item()
                nupdate += 1

        # Compute average stats
        pi_loss_epoch /= nupdate
        vf_loss_epoch /= nupdate
        entropy_epoch /= nupdate
        skill_pi_loss_epoch /= skill_nupdate
        skill_vf_loss_epoch /= skill_nupdate
        skill_entropy_epoch /= skill_nupdate

        # Define train stats
        train_stats = {
            "pi_loss": pi_loss_epoch,
            "vf_loss": vf_loss_epoch,
            "entropy": entropy_epoch,
            "skill_pi_loss": skill_pi_loss_epoch,
            "skill_vf_loss": skill_vf_loss_epoch,
            "skill_entropy": skill_entropy_epoch,
        }

        # Increase PPO count
        self.ppo_count += 1

        if self.ppo_count % self.aux_freq == 0:
            # Copy model and set it to eval mode
            old_model = copy.deepcopy(self.model)
            # /usr/local/lib/python3.10/dist-packages/torch/nn/modules/rnn.py:881: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at ../aten/src/ATen/native/cudnn/RNN.cpp:982.) result = _VF.lstm(input, batch_sizes, hx, self._flat_weights, self.bias,
            old_model.rnn.layer.flatten_parameters()
            old_model.eval()

            # Run aux phase
            aux_loss_epoch = 0
            pi_dist_epoch = 0
            vf_dist_epoch = 0
            skill_pi_dist_epoch = 0
            skill_vf_dist_epoch = 0
            aux_nupdate = 0

            # TODO: reset
            # TODO: print buffer size
            for _ in range(self.aux_nepoch):
                # pad data
                masks = th.nn.utils.rnn.pad_sequence(
                    [th.ones(len(traj), device=self.device) for traj in self.buffer.aux_skill_policy_data],
                    padding_value=0,
                    batch_first=True,
                )
                # NOTE: long or float
                assert masks.ndim <=3
                lengths = th.sum(masks, dim=1).long().to("cpu")
                if lengths.ndim > 1:
                    lengths = lengths.squeeze(-1)

                self.buffer.aux_skill_policy_data = th.nn.utils.rnn.pad_sequence(
                    self.buffer.aux_skill_policy_data, padding_value=0, batch_first=True
                )
                data_loader = self.buffer.aux_skill_policy_data_loader(self.aux_nbatch)
                for indices in data_loader:
                    losses = self.model.compute_aux_losses(
                        self.buffer.aux_skill_policy_data[indices],
                        lengths[indices],
                        masks[indices],
                        old_model,
                    )

                    loss = (
                        losses["rec_loss"]
                        + +self.pi_dist_coef * losses["pi_dist"] * 0.5
                        + self.vf_dist_coef * losses["vf_dist"] * 0.5
                        + self.pi_dist_coef * losses["skill_pi_dist"] * 0.5
                        + self.vf_dist_coef * losses["skill_vf_dist"] * 0.5
                    )

                    # Update parameters
                    self.aux_optimizer.zero_grad()
                    # TODO!: Why works
                    self.model.train()
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.aux_optimizer.step()

                    # Update stats
                    aux_loss_epoch += losses["rec_loss"].item()
                    pi_dist_epoch += losses["pi_dist"].item()
                    vf_dist_epoch += losses["vf_dist"].item()
                    skill_pi_dist_epoch += losses["skill_pi_dist"].item()
                    skill_vf_dist_epoch += losses["skill_vf_dist"].item()
                    aux_nupdate += 1
            # Compute average stats
            aux_loss_epoch /= aux_nupdate
            pi_dist_epoch /= aux_nupdate
            vf_dist_epoch /= aux_nupdate
            skill_pi_dist_epoch /= aux_nupdate
            skill_vf_dist_epoch /= aux_nupdate

            # Define aux train stats
            aux_train_stats = {
                "match_loss": aux_loss_epoch,
                "pi_dist": pi_dist_epoch,
                "vf_dist": vf_dist_epoch,
                "skill_pi_dist": skill_pi_dist_epoch,
                "skill_vf_dist": skill_vf_dist_epoch,
            }

            # Update train stats
            train_stats.update(aux_train_stats)

            self.buffer.reset_aux_data()

        # update selected skill traj's priority
        # NOTE: some may have been removed
        min_id = self.model.pbuffer.min_id()
        for id in th.unique(storage.traj_ids):
            assert id >= -1
            if id != -1 and id >= min_id:
                traj = self.model.pbuffer.trajs[id - min_id]
                assert len(traj)-1>=self.min_skill_len
                rec_error = self.model.compute_rec_error(traj)
                self.model.pbuffer.update(id - min_id, **{"rec_errors": rec_error})

        self.buffer.reset_policy_data()

        return train_stats
