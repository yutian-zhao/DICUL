from typing import Dict
import random

import torch as th

from gym import spaces

from dicul.model.base import BaseModel
from dicul.impala_cnn import ImpalaCNN
from dicul.action_head import CategoricalActionHead
from dicul.mse_head import ScaledMSEHead
from dicul.torch_util import FanInInitReLULayer, RNN
from dicul.vqvae import VQVAE
from dicul.prioritized_buffer import (
    PrioritizedBuffer,
    PriorityTraj,
)


class PPODICULModel(BaseModel):
    """
    Define PPOModel architechture.
    Function act(), encode(), compute_losses()
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        device: th.device,
        impala_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
        action_head_kwargs: Dict = {},
        mse_head_kwargs: Dict = {},
        vqvae_kwargs: Dict = {},
        max_skill_len: Dict = {},
    ):
        super().__init__(observation_space, action_space)

        # Encoder
        self.obs_shape = getattr(self.observation_space, "shape")
        self.enc = ImpalaCNN(
            self.obs_shape,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **impala_kwargs,
        )
        self.outsize = impala_kwargs["outsize"]

        self.rnn = RNN(self.outsize, self.outsize, init_scale=1.4)

        self.vqvae = VQVAE(insize=self.outsize, outsize=self.outsize, **vqvae_kwargs)

        # Before Heads
        self.linear = FanInInitReLULayer(
            self.outsize,
            hidsize,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )
        # NOTE: input is the concatenation of features and skills
        self.linear_skill = FanInInitReLULayer(
            self.outsize + vqvae_kwargs["D"],
            hidsize,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )
        self.hidsize = hidsize

        # Heads
        self.num_skills = vqvae_kwargs["K"]
        self.pi_head = CategoricalActionHead(
            insize=hidsize,
            num_actions=self.num_skills,
            **action_head_kwargs,
        )
        self.vf_head = ScaledMSEHead(
            insize=hidsize,
            outsize=1,
            **mse_head_kwargs,
        )

        self.num_actions = getattr(self.action_space, "n")
        self.pi_skill_head = CategoricalActionHead(
            insize=hidsize,
            num_actions=self.num_actions,
            **action_head_kwargs,
        )
        self.vf_skill_head = ScaledMSEHead(
            insize=hidsize,
            outsize=1,
            **mse_head_kwargs,
        )

        self.max_skill_len = max_skill_len
        self.pbuffer = PrioritizedBuffer()
        # self.last_h = th.zeros((1, outsize))
        # self.last_c = th.zeros((1, outsize))
        # self.last_skill_hs = th.zeros((1, outsize))
        # self.last_skill_cs = th.zeros((1, outsize))
        self.device = device

    @th.no_grad()
    def if_terminate(self, rewards: th.Tensor, skill_step_counts, masks: th.Tensor):
        # all (N, D), return bool tensor
        if_init_state = masks.long() == 0
        if_reach_max_length = skill_step_counts >= self.max_skill_len
        # if th.any(if_reach_max_length):
        #     print(if_reach_max_length)
        # NOTE: This holds only for a single step
        if_unlock_achievement = rewards > 0.1
        a = th.logical_or(if_init_state, if_reach_max_length)
        b = th.logical_or(
            a,
            if_unlock_achievement,
        )
        return th.logical_or(
            a,
            b,
        ).reshape(-1)

    @th.no_grad()
    def act(
        self,
        obs: th.Tensor,
        rewards: th.Tensor,
        skills: th.Tensor,
        skill_step_counts: th.Tensor,
        traj_ids: th.Tensor,
        masks: th.Tensor,
        hs: th.Tensor,
        cs: th.Tensor,
        skill_hs: th.Tensor,
        skill_cs: th.Tensor,
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        # Check training mode
        assert not self.training

        # Pass through model
        # TODO!: check squeeze
        outputs = self.forward(obs=obs, hs=hs, cs=cs, **kwargs)
        outputs = {
            k: (
                v.squeeze(1)
                if len(v.shape) > 2
                and k
                not in [
                    "features",
                    "hs",
                    "cs",
                    # 'vpreds',
                    # 'skill_vpreds',
                    "skill_recurrent_features",
                    "recurrent_features",
                    "skill_cs",
                    "skill_hs",
                ]
                else v
            )
            for k, v in outputs.items()
        }
        # Sample skills
        # TODO: Check squeeze. Sample skills when skills terminate.
        # TODO: Extract skill trajectories and episodes on the go. Might be more efficient.
        pi_logits = outputs["pi_logits"]
        termination = self.if_terminate(
            rewards=rewards,
            skill_step_counts=skill_step_counts,
            masks=masks,
        )
        # .detach().clone()
        skills = skills.clone()
        # NOTE: successive reward happen
        # assert not th.any(skill_step_counts[termination]==1), f"rewards: {rewards} \n masks: {masks}"
        skill_step_counts = skill_step_counts.clone() + 1  # increment counts
        traj_ids = traj_ids.clone()
        if th.any(termination):
            new_skills = self.pi_head.sample(pi_logits[termination])
            skills[termination] = new_skills
            skill_step_counts[termination] = 1
            traj_ids[termination] = -1
            # epsilon = random.uniform(0, 1)
            if_sample = (th.rand_like(termination.float()) * termination.int()) > 0.8
            num_sample = th.sum(if_sample.int())
            if th.any(if_sample) and len(self.pbuffer) >= num_sample:
                # TODO: should be tensor
                # TODO: should update priority before sampling
                sample_indices, sample_ids = self.pbuffer.sample(num_sample)
                sample_trajs = []
                for ind in sample_indices:
                    sample_trajs.append(self.pbuffer.trajs[ind])
                masks = th.nn.utils.rnn.pad_sequence(
                    [th.ones(len(traj)) for traj in sample_trajs],
                    padding_value=0,
                    batch_first=True,
                )
                # NOTE: long or float
                assert masks.ndim <=3
                lengths = th.sum(masks, dim=1).long().to("cpu")
                if lengths.ndim > 1:
                    lengths = lengths.squeeze(-1)
                sample_trajs = th.nn.utils.rnn.pad_sequence(sample_trajs, padding_value=0, batch_first=True)
                _, recurrent_features, _, _ = self.compute_features(sample_trajs, lengths)
                index_shape = [*recurrent_features.shape]
                index_shape[1] = 1
                index = (lengths - 1).reshape(-1, *(1,)*(len(index_shape)-1))
                index = index.expand(*index_shape).long().to(self.device)
                recurrent_features = th.gather(
                    recurrent_features, dim=1, index=index
                )
                indices = self.vqvae.encode(recurrent_features, just_indices=True)
                skills[if_sample] = indices
                traj_ids[if_sample] = th.tensor(sample_ids, device=self.device).unsqueeze(-1).long()

        # Sample skills' actions
        skill_policy_outputs = self.skill_policy_forward(
            obs=obs,
            features=outputs["features"],
            skills=skills.unsqueeze(1),  # nproc, 1 -> nproc, L, 1
            termination=termination,
            skill_hs=skill_hs,
            skill_cs=skill_cs,
        )
        skill_policy_outputs = {
            k: (
                v.squeeze(1)
                if len(v.shape) > 2
                and k
                not in [
                    "features",
                    "hs",
                    "cs",
                    # 'vpreds',
                    # 'skill_vpreds',
                    "skill_recurrent_features",
                    "recurrent_features",
                    "skill_cs",
                    "skill_hs",
                ]
                else v
            )
            for k, v in skill_policy_outputs.items()
        }
        outputs.update(skill_policy_outputs)
        skill_pi_logits = outputs["skill_pi_logits"]
        actions = self.pi_skill_head.sample(skill_pi_logits)

        # Compute log probs
        log_probs = self.pi_head.log_prob(pi_logits, skills)
        skill_log_probs = self.pi_skill_head.log_prob(skill_pi_logits, actions)

        # Denormalize vpreds
        vpreds = outputs["vpreds"]
        vpreds = self.vf_head.denormalize(vpreds)
        skill_vpreds = outputs["skill_vpreds"]
        skill_vpreds = self.vf_skill_head.denormalize(skill_vpreds)
        # TODO: Check policy and skill.

        # Update outputs
        outputs.update(
            {
                "skills": skills,
                "log_probs": log_probs,
                "vpreds": vpreds,
                "actions": actions,
                "skill_log_probs": skill_log_probs,
                "skill_vpreds": skill_vpreds,
                "traj_ids": traj_ids,
                "skill_step_counts": skill_step_counts,
            }
        )
        # TODO: check shape
        outputs = {
            k: (
                v.squeeze(1)
                if len(v.shape) > 2
                and k
                not in [
                    "features",
                    "hs",
                    "cs",
                    # 'vpreds',
                    # 'skill_vpreds',
                    "skill_recurrent_features",
                    "recurrent_features",
                    "skill_cs",
                    "skill_hs",
                ]
                else v
            )
            for k, v in outputs.items()
        }
        return outputs

    def compute_features(
        self, trajs: th.Tensor, lengths: th.Tensor = None, hs: th.Tensor = None, cs: th.Tensor = None
    ):
        ntrajs = len(trajs)
        # TODO: check shape
        trajs = trajs.reshape((-1, *self.obs_shape))
        features = self.enc(trajs)
        features = features.reshape((ntrajs, -1, self.outsize))  # (N,D) -> (N,L,D)
        if hs is not None and len(features.shape) > len(hs.shape):
            hs = hs.unsqueeze(0)
            cs = cs.unsqueeze(0)
        if lengths is not None:
            # 'lengths' argument should be a 1D CPU int64 tensor,
            # NOTE: this will squeeze the dim to max lengths
            packed_trajs = th.nn.utils.rnn.pack_padded_sequence(
                features, lengths, batch_first=True, enforce_sorted=False
            )  # (B, T, C, D)
            packed, (hs_, cs_) = self.rnn(packed_trajs, hs, cs)  # (N,L,D) -> (N,L,D)
            recurrent_features, lengths = th.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        else:
            recurrent_features, (hs_, cs_) = self.rnn(features, hs, cs)
        return features, recurrent_features, hs_, cs_

    def forward(
        self, obs: th.Tensor, lengths: th.Tensor = None, hs: th.Tensor = None, cs: th.Tensor = None, **kwargs
    ) -> Dict[str, th.Tensor]:
        # output (N,L,D)
        # Pass through encoder
        # TODO: handle n step sequences
        # ntrajs = len(obs)
        # obs = obs.reshape((-1, *self.obs_shape))
        # features = self.enc(obs).reshape((ntrajs, -1, *self.obs_shape))  # (N,D) -> (N,L,D)
        # recurrent_features, (hs, cs) = self.rnn(features, hs, cs)  # (N,L,D) -> (N,L,D)
        features, recurrent_features, hs, cs = self.compute_features(obs, lengths, hs, cs)
        # TODO: check squeeze
        latents = self.linear(recurrent_features)

        # Pass through heads
        pi_latents = vf_latents = latents
        pi_logits = self.pi_head(pi_latents)
        vpreds = self.vf_head(vf_latents)

        # Define outputs
        outputs = {
            "features": features,  # NOTE: detach
            "recurrent_features": recurrent_features,
            # "latents": latents,
            # "pi_latents": pi_latents,
            # "vf_latents": vf_latents,
            "pi_logits": pi_logits,
            "vpreds": vpreds,
            "hs": hs.squeeze(0) if len(hs.shape) > 2 else hs,
            "cs": cs.squeeze(0) if len(cs.shape) > 2 else cs,
        }

        return outputs

    def skill_policy_forward(
        self,
        obs: th.Tensor,
        skills: th.Tensor,
        lengths: th.Tensor = None,
        skill_hs: th.Tensor = None,
        skill_cs: th.Tensor = None,
        features: th.Tensor = None,
        skill_recurrent_features: th.Tensor = None,
        termination: th.Tensor = None,
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        # output (N,L,D)
        # if not features and not termination:
        # NOTE: the desire shape of termination is (N, D), is not None only when acting
        if skill_recurrent_features is None:
            if termination is not None:
                not_terminated = th.logical_not(termination).int()
                # reset hidden states if terminated
                if skill_hs is not None and len(features.shape) > len(skill_hs.shape):
                    skill_hs = (skill_hs * (not_terminated.unsqueeze(-1))).unsqueeze(0)
                    skill_cs = (skill_cs * (not_terminated.unsqueeze(-1))).unsqueeze(0)
                skill_recurrent_features, (skill_hs, skill_cs) = self.rnn(
                    features, skill_hs, skill_cs
                )  # (N,L,D) -> (N,L,D)
            else:
                # ntrajs = len(obs)
                # obs = obs.reshape((-1, *self.obs_shape))
                # features = self.enc(obs).reshape((ntrajs, -1, *self.obs_shape))  # (N,D) -> (N,L,D)
                features, skill_recurrent_features, skill_hs, skill_cs = self.compute_features(
                    obs, lengths, skill_hs, skill_cs
                )

        # NOTE: surpport N, L
        # NOTE: detach
        # NOTE: check squeeze
        skill_policy_latents = self.linear_skill(
            th.cat(
                [skill_recurrent_features, self.vqvae.book.book.weight.data[skills.squeeze(-1)].detach()],
                axis=-1,
            )
        )
        pi_skill_latents = vf_skill_latents = skill_policy_latents
        skill_pi_logits = self.pi_skill_head(pi_skill_latents)
        skill_vpreds = self.vf_skill_head(vf_skill_latents)

        outputs = {
            "skill_recurrent_features": skill_recurrent_features,
            # "skill_policy_latents": skill_policy_latents,
            # "pi_skill_latents": pi_skill_latents,
            # "vf_skill_latents": vf_skill_latents,
            "skill_pi_logits": skill_pi_logits,
            "skill_vpreds": skill_vpreds,
            "skill_hs": skill_hs.squeeze(0) if skill_hs is not None and len(skill_hs.shape) > 2 else skill_hs,
            "skill_cs": skill_cs.squeeze(0) if skill_cs is not None and len(skill_cs.shape) > 2 else skill_cs,
        }

        return outputs

    def encode(self, obs: th.Tensor) -> th.Tensor:
        pass

    # def encode(self, obs: th.Tensor, hs: th.Tensor, cs: th.Tensor) -> th.Tensor:
    #     # obs (N(n_proc), D)
    #     # Pass through encoder
    #     x = self.enc(obs).unsqueeze(1)  # (N,D) -> (N,L,D)
    #     x, (hs, cs) = self.rnn(x, hs, cs)  # (N,L,D) -> (N,L,D)

    #     return x, (hs, cs)

    # def traj_encode(self, obs: th.Tensor, hs: th.Tensor, cs: th.Tensor) -> th.Tensor:
    #     # obs (L(n_steps), D)
    #     # Pass through encoder
    #     x = self.enc(obs).unsqueeze(0) # (L,D) -> (N,L,D)
    #     x, (hs, cs)= self.rnn(x, hs, cs) # (N,L,D) -> (N,L,D)

    #     return x, (hs, cs)
    @th.no_grad()
    def compute_intrinsic_reward(
        self,
        mode: str,
        skill_recurrent_features: th.Tensor,
        skill_hs: th.Tensor,
        skill_cs: th.Tensor,
        next_obs: th.Tensor,
        skills: th.Tensor,
        **kwargs,
    ):
        # NOTE: skills are zs, not used in 'master' mode
        # self.eval()
        _, next_recurrent_features, _, _ = self.compute_features(next_obs, hs=skill_hs, cs=skill_cs)
        master_intrinsic_reward = None
        skill_intrinsic_reward = None
        if mode == "master" or mode == "both":
            _, prob = self.vqvae.loss(
                next_recurrent_features,
                skill_recurrent_features,
                next_recurrent_features - skill_recurrent_features,
                if_prob=True,
            )
            # lower the prob, higher the reconstruction error, higher reward
            master_intrinsic_reward = 1 - prob
            if mode == "master":
                return master_intrinsic_reward
        if mode == "skill" or mode == "both":
            # TODO: check zs[skill] and others to squeeze
            _, prob = self.vqvae.decoder_loss(
                skills,
                skill_recurrent_features,
                next_recurrent_features - skill_recurrent_features,
                if_prob=True,
            )
            skill_intrinsic_reward = prob
            if mode == "skill":
                return skill_intrinsic_reward
        return (master_intrinsic_reward, skill_intrinsic_reward)

    def masked_average(
        self,
        input: th.Tensor,
        masks: th.Tensor,
    ):
        return th.sum(input * masks) / th.sum(masks)

    def compute_losses(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        log_probs: th.Tensor,
        vtargs: th.Tensor,
        advs: th.Tensor,
        masks: th.Tensor,
        mode: str,
        # lengths: th.Tensor,
        skills: th.Tensor = None,
        # features:th.Tensor=None,
        clip_param: float = 0.2,
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        # NOTE: float or long
        assert masks.ndim <=3
        lengths = th.sum(masks, dim=1).long().to("cpu")
        if lengths.ndim > 1:
            lengths = lengths.squeeze(-1)
        max_len = th.max(lengths)
        advs = advs[:, :max_len]
        masks = masks[:, :max_len]
        actions = actions[:, :max_len]
        vtargs = vtargs[:, :max_len]
        log_probs = log_probs[:, :max_len]
        if skills is not None:
            skills = skills[:, :max_len]
        # Pass through model
        if mode == "master":
            outputs = self.forward(obs, lengths, **kwargs)  # (B,L,D)
            policy_head = self.pi_head
            value_head = self.vf_head
            pi_logits = outputs["pi_logits"]
            vpreds = outputs["vpreds"]
        elif mode == "skill" and skills is not None:
            outputs = self.skill_policy_forward(obs, skills, lengths=lengths, **kwargs)  # (B,L,D)
            policy_head = self.pi_skill_head
            value_head = self.vf_skill_head
            pi_logits = outputs["skill_pi_logits"]
            vpreds = outputs["skill_vpreds"]

        # Compute policy loss
        new_log_probs = policy_head.log_prob(pi_logits, actions)
        ratio = th.exp(new_log_probs - log_probs)
        ratio_clipped = th.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        pi_loss = self.masked_average(-th.min(advs * ratio, advs * ratio_clipped), masks)

        # Compute entropy
        entropy = self.masked_average(policy_head.entropy(pi_logits), masks)

        # Compute value loss
        vf_loss = self.masked_average(value_head.mse_loss(vpreds, vtargs), masks)

        # Define losses
        losses = {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}

        return losses

    def compute_aux_losses(self, trajs, lengths, masks, old_model):
        # TODO: add length to forward
        # TODO!: buffer should be moved out of model
        max_len = th.max(lengths)
        masks = masks[:, :max_len]
        masks = masks.unsqueeze(-1)  # N,L,1
        outputs = self.forward(trajs, lengths)  # N,L,D
        with th.no_grad():
            old_outputs = old_model.forward(trajs, lengths)
        # features, recurrent_features, hs, cs = self.compute_features(trajs, lengths, hs, cs)
        recurrent_features = outputs["recurrent_features"]
        old_recurrent_features = old_outputs["recurrent_features"]
        # NOTE: sample skills instead of used skills
        sampled_skills = th.randint(
            low=0,
            high=self.num_skills,
            size=(recurrent_features.shape[0], recurrent_features.shape[1]),
        )
        skill_outputs = self.skill_policy_forward(
            trajs, sampled_skills, lengths=lengths, skill_recurrent_features=recurrent_features
        )
        with th.no_grad():
            old_skill_outputs = old_model.skill_policy_forward(
                trajs, sampled_skills, lengths=lengths, skill_recurrent_features=old_recurrent_features
            )

        rec_loss, _ = self.vqvae.loss(
            recurrent_features[:, 1:],
            recurrent_features[:, :-1],
            (recurrent_features[:, 1:] - recurrent_features[:, :-1]).detach(),
            reduction="none",
        )
        rec_loss = self.masked_average(rec_loss, masks[:, 1:])
        # TODO: for loop for each recurrent features
        # compute kl regularizer

        pi_dist = self.masked_average(
            self.pi_head.kl_divergence(outputs["pi_logits"], old_outputs["pi_logits"]), masks
        )
        skill_pi_dist = self.masked_average(
            self.pi_skill_head.kl_divergence(
                skill_outputs["skill_pi_logits"], old_skill_outputs["skill_pi_logits"]
            ),
            masks,
        )
        vf_dist = self.masked_average(
            (self.vf_head.mse_loss(outputs["vpreds"], old_outputs["vpreds"])), masks
        )
        skill_vf_dist = self.masked_average(
            (self.vf_skill_head.mse_loss(skill_outputs["skill_vpreds"], old_skill_outputs["skill_vpreds"])),
            masks,
        )

        losses = {
            "rec_loss": rec_loss,
            "pi_dist": pi_dist,
            "skill_pi_dist": skill_pi_dist,
            "vf_dist": vf_dist,
            "skill_vf_dist": skill_vf_dist,
        }

        return losses

    # TODO!: differences
    @th.no_grad()
    def compute_rec_error(self, traj):
        # single traj with no padding L, D
        # self.eval()
        traj = traj.unsqueeze(0)
        outputs = self.forward(traj)
        recurrent_features = outputs["recurrent_features"]
        rec_loss, _ = self.vqvae.loss(
            recurrent_features[:, 1:],
            recurrent_features[:, :-1],
            (recurrent_features[:, 1:] - recurrent_features[:, :-1]).detach(),
            reduction="none",
        )
        return th.max(rec_loss)
