from typing import Dict, Iterator

import torch as th
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from gym import spaces

from dicul.model.base import BaseModel


class RolloutStorage:
    """
    RolloutStorage serves as RL algorithm's buffer. It will be reset or overwritten every rollout and update.
    Functions: insert(), reset(), compute_returns(), get_data_loader()
    """

    def __init__(
        self,
        nstep: int,
        nproc: int,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        outsize: int,
        device: th.device,
    ):
        # Params
        self.nstep = nstep
        self.nproc = nproc
        self.device = device

        # Get obs shape and action dim
        assert isinstance(observation_space, spaces.Box)
        assert isinstance(action_space, spaces.Discrete), type(action_space)
        obs_shape = getattr(observation_space, "shape")
        action_shape = (1,)

        # Tensors
        """
            obs
            masks
            successes
            skill_step_counts 
            traj_ids
            timesteps

            rewards
            master_intrinsic_rewards
            skill_intrinsic_rewards
            returns
            advs

            # "features": features,
            # "recurrent_features": recurrent_features,
            # "skill_recurrent_features": skill_recurrent_features,

            # "pi_logits": pi_logits,
            vpreds
            skills 
            log_probs
            hs (only last)
            cs (only last)
            
            # "pi_skill_logits": pi_skill_logits,
            skill_vpreds
            actions
            skill_log_probs
            skill_hs (only last)
            skill_cs (only last)
            
        """
        # TODO: Replace hs and cs buffer*2 with a keeper
        self.obs = th.zeros(nstep + 1, nproc, *obs_shape, device=device)
        self.masks = th.ones(nstep + 1, nproc, 1, device=device)
        self.successes = th.zeros(nstep + 1, nproc, 22, device=device).long()
        self.timesteps = th.zeros(nstep + 1, nproc, 1, device=device).long()
        # NOTE: t steps, default to 0
        self.skill_step_counts = th.zeros(nstep+1, nproc, 1, device=device).long()
        # NOTE: default is -1, traj to imitate, count id
        self.traj_ids = -th.ones(nstep, nproc, 1, device=device).long()

        # NOTE: This refers to extrinsic rewards
        self.rewards = th.zeros(nstep, nproc, 1, device=device)
        self.master_intrinsic_rewards = th.zeros(nstep, nproc, 1, device=device)
        self.skill_intrinsic_rewards = th.zeros(nstep, nproc, 1, device=device)
        # NOTE: return to contain the sum of intrinsic and extrinsic rewards
        self.returns = th.zeros(nstep, nproc, 1, device=device)
        self.advs = th.zeros(nstep, nproc, 1, device=device)
        self.skill_returns = th.zeros(nstep, nproc, 1, device=device)
        self.skill_advs = th.zeros(nstep, nproc, 1, device=device)

        # NOTE: t steps, default is -1, assert not -1
        self.skills = -th.ones(nstep, nproc, 1, device=device).long()
        self.actions = th.zeros(nstep, nproc, *action_shape, device=device).long()
        self.vpreds = th.zeros(nstep + 1, nproc, 1, device=device)
        self.skill_vpreds = th.zeros(nstep + 1, nproc, 1, device=device)
        self.log_probs = th.zeros(nstep, nproc, 1, device=device)
        self.skill_log_probs = th.zeros(nstep, nproc, 1, device=device)
        self.hs = th.zeros(nstep, nproc, outsize, device=device)
        self.cs = th.zeros(nstep, nproc, outsize, device=device)
        self.skill_hs = th.zeros(nstep, nproc, outsize, device=device)
        self.skill_cs = th.zeros(nstep, nproc, outsize, device=device)

        # Step
        self.step = 0

    def __getitem__(self, key: str) -> th.Tensor:
        return getattr(self, key)

    def get_inputs(self, step: int):
        inputs = {
            "obs": self.obs[step],
            "masks": self.masks[step],
            "rewards": self.rewards[step - 1],
            "skill_step_counts": self.skill_step_counts[step - 1] if step else self.skill_step_counts[-2], # NOTE: skill_step_counts[-1] is the current step
            "traj_ids": self.traj_ids[step - 1],
            "skills": self.skills[step - 1],
            "hs": self.hs[step - 1],
            "cs": self.cs[step - 1],
            "skill_hs": self.skill_hs[step - 1],
            "skill_cs": self.skill_cs[step - 1],
        }
        return inputs

    def insert(
        self,
        obs: th.Tensor,
        masks: th.Tensor,
        successes: th.Tensor,
        skill_step_counts: th.Tensor,
        traj_ids: th.Tensor,
        rewards: th.Tensor,
        master_intrinsic_rewards: th.Tensor,
        skill_intrinsic_rewards: th.Tensor,
        vpreds: th.Tensor,
        skills: th.Tensor,
        log_probs: th.Tensor,
        hs: th.Tensor,
        cs: th.Tensor,
        skill_vpreds: th.Tensor,
        actions: th.Tensor,
        skill_log_probs: th.Tensor,
        skill_hs: th.Tensor,
        skill_cs: th.Tensor,
        **kwargs,
    ):
        # Get prev successes, timesteps, and states
        # prev_successes = self.successes[self.step]
        # prev_states = self.states[self.step]
        prev_timesteps = self.timesteps[self.step]

        # Update timesteps
        timesteps = prev_timesteps + 1

        # Update states if new achievment is unlocked
        # success_conds = successes != prev_successes
        # success_conds = success_conds.any(dim=-1, keepdim=True)
        # if success_conds.any():
        #     with th.no_grad():
        #         next_latents = model.encode(obs)
        #     states = next_latents - latents
        #     states = F.normalize(states, dim=-1)
        #     states = th.where(success_conds, states, prev_states)
        # else:
        #     states = prev_states

        # Update successes, timesteps, and states if done
        done_conds = masks == 0
        successes = th.where(done_conds, 0, successes)
        timesteps = th.where(done_conds, 0, timesteps)
        # states = th.where(done_conds, 0, states)

        # Update tensors
        self.obs[self.step + 1].copy_(obs)
        self.masks[self.step + 1].copy_(masks)
        self.successes[self.step + 1].copy_(successes)
        self.timesteps[self.step + 1].copy_(timesteps)
        self.skill_step_counts[self.step].copy_(skill_step_counts)
        # NOTE: consider step+1
        self.traj_ids[self.step].copy_(traj_ids)

        self.rewards[self.step].copy_(rewards)
        self.master_intrinsic_rewards[self.step].copy_(master_intrinsic_rewards)
        self.skill_intrinsic_rewards[self.step].copy_(skill_intrinsic_rewards)

        self.skills[self.step].copy_(skills)
        self.actions[self.step].copy_(actions)
        self.vpreds[self.step].copy_(vpreds)
        self.skill_vpreds[self.step].copy_(skill_vpreds)
        self.log_probs[self.step].copy_(log_probs)
        self.skill_log_probs[self.step].copy_(skill_log_probs)
        self.hs[self.step].copy_(hs)
        self.cs[self.step].copy_(cs)
        self.skill_hs[self.step].copy_(skill_hs)
        self.skill_cs[self.step].copy_(skill_cs)

        # Update step
        self.step = (self.step + 1) % self.nstep

    def reset(self):
        # Reset tensors
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.successes[0].copy_(self.successes[-1])
        self.timesteps[0].copy_(self.timesteps[-1])

        # Reset step
        self.step = 0

    def compute_returns(self, gamma: float, gae_lambda: float):
        # Compute returns
        # TODO: intrinsic reward beta
        # DEBUG
        beta = 1
        gae = 0
        skill_gae = 0
        skill_masks = self.skill_step_counts== 1 # check no successive terminations
        for step in reversed(range(self.rewards.shape[0])):
            delta = (
                self.rewards[step]
                + beta * self.master_intrinsic_rewards[step]
                + gamma * self.vpreds[step + 1] * self.masks[step + 1]
                - self.vpreds[step]
            )
            skill_delta = (
                self.rewards[step]
                + beta * self.skill_intrinsic_rewards[step]
                + gamma * self.skill_vpreds[step + 1] * skill_masks[step + 1]
                - self.skill_vpreds[step]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            skill_gae = skill_delta + gamma * gae_lambda * skill_masks[step + 1] * skill_gae
            self.returns[step] = gae + self.vpreds[step]
            self.advs[step] = gae
            self.skill_returns[step] = skill_gae + self.skill_vpreds[step]
            self.skill_advs[step] = skill_gae

        # Compute advantages
        self.advs = (self.advs - self.advs.mean()) / (self.advs.std() + 1e-8)
        self.skill_advs = (self.skill_advs - self.skill_advs.mean()) / (self.skill_advs.std() + 1e-8)

    # Not used
    # def get_data_loader(self, nbatch: int) -> Iterator[Dict[str, th.Tensor]]:
    #     # Get sampler
    #     ndata = self.nstep * self.nproc
    #     assert ndata >= nbatch
    #     batch_size = ndata // nbatch
    #     sampler = SubsetRandomSampler(range(ndata))
    #     sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

    #     # Sample batch
    #     obs = self.obs[:-1].view(-1, *self.obs.shape[2:])
    #     # states = self.states[:-1].view(-1, *self.states.shape[2:])
    #     actions = self.actions.view(-1, *self.actions.shape[2:])
    #     vtargs = self.returns.view(-1, *self.returns.shape[2:])
    #     log_probs = self.log_probs.view(-1, *self.log_probs.shape[2:])
    #     advs = self.advs.view(-1, *self.advs.shape[2:])

    #     for indices in sampler:
    #         batch = {
    #             "obs": obs[indices],
    #             # "states": states[indices],
    #             "actions": actions[indices],
    #             "vtargs": vtargs[indices],
    #             "log_probs": log_probs[indices],
    #             "advs": advs[indices],
    #         }
    #         yield batch
