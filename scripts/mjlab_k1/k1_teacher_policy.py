"""K1 teacher policy wrapper for student distillation."""

import torch

from neural_wbc.student_policy.teacher_policy import TeacherPolicy


class K1TeacherPolicy(TeacherPolicy):
    """Wraps a loaded mjlab PPO inference policy as a TeacherPolicy."""

    def __init__(self, inference_fn, checkpoint_path: str):
        super().__init__()
        self._inference_fn = inference_fn
        self.path = checkpoint_path

    def load(self, path, **kwargs):
        raise NotImplementedError("Load via MotionTrackingOnPolicyRunner, then pass inference_fn.")

    @torch.inference_mode()
    def act_rollout(self, observations: torch.Tensor) -> torch.Tensor:
        # RSL-RL MLPModel expects a dict keyed by obs group name ("actor").
        return self._inference_fn({"actor": observations})

    @torch.inference_mode()
    def act(self, observations: torch.Tensor) -> torch.Tensor:
        return self._inference_fn({"actor": observations})
