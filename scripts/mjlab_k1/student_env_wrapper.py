"""Env wrapper exposing teacher + student observations for K1 distillation."""

from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.mdp.commands import MotionCommand

from neural_wbc.core.mask import create_mask

from .k1_student_obs import (
    K1_DISTILL_MASK_MODES,
    K1_MASK_ELEMENT_NAMES,
    compute_k1_student_obs,
)


class K1StudentEnvWrapper:
    """Wraps K1 mjlab env to expose teacher + student obs for distillation.

    Implements the interface expected by StudentPolicyTrainer:
      - get_full_observations() → {"teacher_policy": [N, 125], "student_policy": [N, 222]}
      - step(actions) → (None, None, rew, dones, extras) where extras["observations"] is the above
      - num_envs, device
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        env_wrapped: RslRlVecEnvWrapper,
        mask_modes: dict | None = None,
        enable_sparsity_randomization: bool = False,
    ):
        self._env = env
        self._wrapped = env_wrapped
        self._mask_modes = mask_modes or K1_DISTILL_MASK_MODES
        self._sparsity = enable_sparsity_randomization

        self.num_envs = env.num_envs
        self.device = env.device

        self._motion_cmd = cast(
            MotionCommand,
            env.command_manager.get_term("motion"),
        )
        self._robot = env.scene.entities["robot"]
        self._last_actions = torch.zeros(self.num_envs, 22, device=self.device)
        self._mask = self._sample_mask(self.num_envs)

    def _sample_mask(self, n: int) -> torch.Tensor:
        return create_mask(
            n,
            K1_MASK_ELEMENT_NAMES,
            self._mask_modes,
            self._sparsity,
            self.device,
        ).float()

    def _compute_student_obs(self) -> torch.Tensor:
        obs_dict = compute_k1_student_obs(
            self._motion_cmd,
            self._robot,
            self._last_actions,
            self._mask,
            self.device,
        )
        return torch.cat(list(obs_dict.values()), dim=-1)   # [N, 222]

    def get_full_observations(self) -> dict[str, torch.Tensor]:
        teacher_obs = self._wrapped.get_observations()["actor"]  # [N, 125]
        student_obs = self._compute_student_obs()                # [N, 222]
        return {
            "teacher_policy": teacher_obs,
            "student_policy": student_obs,
        }

    def step(self, actions: torch.Tensor):
        self._last_actions = actions.clone()
        obs_td, rew, dones, extras = self._wrapped.step(actions)

        # Resample mask for environments that just ended an episode.
        done_ids = dones.nonzero(as_tuple=True)[0]
        if done_ids.numel() > 0:
            self._mask[done_ids] = self._sample_mask(done_ids.numel())

        teacher_obs = obs_td["actor"]           # [N, 125]
        student_obs = self._compute_student_obs()  # [N, 222]

        extras["observations"] = {
            "teacher_policy": teacher_obs,
            "student_policy": student_obs,
        }

        return None, None, rew, dones, extras
