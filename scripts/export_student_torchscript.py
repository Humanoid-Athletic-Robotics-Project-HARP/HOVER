"""Export a trained HOVER student policy to TorchScript for deployment.

Usage:
    python scripts/export_student_torchscript.py \
        --student_path logs/student_h1/<FOLDER> \
        --student_checkpoint model_<ITER>.pt \
        --output h1_hover_student.pt \
        --robot h1
"""

import argparse
import json
import os

import torch
import torch.nn as nn

from neural_wbc.student_policy import StudentPolicy


class StudentInferenceWrapper(nn.Module):
    """Wraps the StudentPolicy's inference path so TorchScript sees a clean forward()."""

    def __init__(self, policy: StudentPolicy):
        super().__init__()
        self.policy_net = policy.policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy_net(obs)


def main():
    parser = argparse.ArgumentParser(description="Export HOVER student policy to TorchScript")
    parser.add_argument("--student_path", type=str, required=True, help="Path to student training folder")
    parser.add_argument("--student_checkpoint", type=str, default=None, help="Checkpoint file (e.g. model_50000.pt)")
    parser.add_argument("--output", type=str, default="h1_hover_student.pt", help="Output .pt TorchScript file")
    parser.add_argument("--robot", type=str, choices=["h1"], default="h1")
    args = parser.parse_args()

    config_path = os.path.join(args.student_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    num_obs = config.get("num_obs", 916)
    num_actions = config.get("num_actions", 19)

    student = StudentPolicy(
        num_obs=num_obs,
        num_actions=num_actions,
        policy_hidden_dims=config.get("policy_hidden_dims", [256, 256, 256]),
        activation=config.get("activation", "elu"),
    )

    if args.student_checkpoint:
        ckpt_path = os.path.join(args.student_path, args.student_checkpoint)
    else:
        pts = sorted([f for f in os.listdir(args.student_path) if f.startswith("model_") and f.endswith(".pt")])
        if not pts:
            raise FileNotFoundError(f"No model checkpoints found in {args.student_path}")
        ckpt_path = os.path.join(args.student_path, pts[-1])
        print(f"Using latest checkpoint: {pts[-1]}")

    print(f"Loading checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    student.load_state_dict(state["model_state_dict"])
    student.eval()

    wrapper = StudentInferenceWrapper(student)
    wrapper.eval()

    example_input = torch.randn(1, num_obs)
    scripted = torch.jit.trace(wrapper, example_input)

    scripted.save(args.output)
    print(f"Exported TorchScript model to: {args.output}")
    print(f"  Input shape:  (batch, {num_obs})")
    print(f"  Output shape: (batch, {num_actions})")


if __name__ == "__main__":
    main()
