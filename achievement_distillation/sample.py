from typing import Dict
import numpy as np

from achievement_distillation.wrapper import VecPyTorch
from achievement_distillation.storage import RolloutStorage
from achievement_distillation.model import PPODICULModel


def sample_rollouts(
    venv: VecPyTorch,
    model: PPODICULModel,
    storage: RolloutStorage,
) -> Dict[str, np.ndarray]:
    # Set model to eval model
    model.eval()

    # Sample rollouts
    episode_rewards = []
    episode_lengths = []
    achievements = []
    successes = []

    for step in range(storage.nstep):
        """
        outputs["obs"] = obs
        outputs["rewards"] = rewards
        outputs["masks"] = 1.0 - dones
        outputs["successes"] = infos["successes"]
        "features": features,
        "recurrent_features": recurrent_features,
        "latents": latents,
        "pi_latents": pi_latents,
        "vf_latents": vf_latents,
        "pi_logits": pi_logits,
        "vpreds": vpreds,
        "skills": skills,
        "log_probs": log_probs,
        "hs": hs,
        "cs": cs,
        "skill_policy_latents": skill_policy_latents,
        "skill_recurrent_features": skill_recurrent_features,
        "pi_skill_latents": pi_skill_latents,
        "vf_skill_latents": vf_skill_latents,
        "pi_skill_logits": pi_skill_logits,
        "skill_vpreds": skill_vpreds,
        "actions": actions,
        "skill_log_probs": skill_log_probs,
        "skill_hs": skill_hs,
        "skill_cs": skill_cs,
        """
        # Pass through model
        inputs = storage.get_inputs(step)
        outputs = model.act(**inputs)
        skills = outputs["skills"]
        actions = outputs["actions"]

        # Step environment
        obs, rewards, dones, infos = venv.step(actions)
        # must be DICUL
        master_intrinsic_reward, skill_intrinsic_reward = model.compute_intrinsic_reward(
            mode="both",
            next_obs=obs,
            skills=outputs["skills"],
            skill_hs=outputs["skill_hs"],
            skill_cs=outputs["skill_cs"],
            skill_recurrent_features=outputs["skill_recurrent_features"], # master policy also uses skill features because encoder should encode skill trajectories
        )
        outputs["obs"] = obs
        outputs["rewards"] = rewards
        outputs["masks"] = 1.0 - dones
        outputs["successes"] = infos["successes"]
        outputs["master_intrinsic_rewards"] = master_intrinsic_reward
        outputs["skill_intrinsic_rewards"] = skill_intrinsic_reward

        # Update storage
        storage.insert(**outputs)

        # Update stats
        for i, done in enumerate(dones):
            if done:
                # Episode lengths
                episode_length = infos["episode_lengths"][i].cpu().numpy()
                episode_lengths.append(episode_length)

                # Episode rewards
                episode_reward = infos["episode_rewards"][i].cpu().numpy()
                episode_rewards.append(episode_reward)

                # Achievements
                achievement = infos["achievements"][i].cpu().numpy()
                achievements.append(achievement)

                # Successes
                success = infos["successes"][i].cpu().numpy()
                successes.append(success)

    # Pass through model
    inputs = storage.get_inputs(step=-1)
    outputs = model.act(**inputs)
    vpreds = outputs["vpreds"]
    skill_vpreds = outputs["skill_vpreds"]
    skill_step_counts = outputs["skill_step_counts"]

    # Update storage
    storage.vpreds[-1].copy_(vpreds)
    storage.skill_vpreds[-1].copy_(skill_vpreds)
    storage.skill_step_counts[-1].copy_(skill_step_counts)

    # Stack stats
    episode_lengths = np.stack(episode_lengths, axis=0).astype(np.int32)
    episode_rewards = np.stack(episode_rewards, axis=0).astype(np.float32)
    achievements = np.stack(achievements, axis=0).astype(np.int32)
    successes = np.stack(successes, axis=0).astype(np.int32)

    # Define rollout stats
    rollout_stats = {
        "episode_lengths": episode_lengths,
        "episode_rewards": episode_rewards,
        "achievements": achievements,
        "successes": successes,
    }

    return rollout_stats
