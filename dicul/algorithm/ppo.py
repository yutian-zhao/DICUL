import torch.nn as nn
import torch.optim as optim

from dicul.model.ppo import PPOModel
from dicul.algorithm.base import BaseAlgorithm
from dicul.storage import RolloutStorage


class PPOAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        model: PPOModel,
        ppo_nepoch: int,
        ppo_nbatch: int,
        clip_param: float,
        vf_loss_coef: float,
        ent_coef: float,
        lr: float,
        max_grad_norm: float,
    ):
        super().__init__(model)
        self.model: PPOModel

        # PPO params
        self.clip_param = clip_param
        self.ppo_nepoch = ppo_nepoch
        self.ppo_nbatch = ppo_nbatch
        self.vf_loss_coef = vf_loss_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def update(self, storage: RolloutStorage):
        # Set model to training mode
        self.model.train()

        # Run PPO
        pi_loss_epoch = 0
        vf_loss_epoch = 0
        entropy_epoch = 0
        nupdate = 0

        for _ in range(self.ppo_nepoch):
            # Get data loader
            data_loader = storage.get_data_loader(self.ppo_nbatch)

            for batch in data_loader:
                # Compute loss
                losses = self.model.compute_losses(**batch, clip_param=self.clip_param)
                pi_loss = losses["pi_loss"]
                vf_loss = losses["vf_loss"]
                entropy = losses["entropy"]
                loss = pi_loss + self.vf_loss_coef * vf_loss - self.ent_coef * entropy

                # Update parameter
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
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

        # Define train stats
        train_stats = {
            "pi_loss": pi_loss_epoch,
            "vf_loss": vf_loss_epoch,
            "entropy": entropy_epoch,
        }

        return train_stats
