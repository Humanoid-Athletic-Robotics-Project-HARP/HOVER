# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional

# TODO: Replace MotionLibH1 with the agnostic version when it's ready.
from phc.utils.motion_lib_h1 import MotionLibH1
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree

from neural_wbc.core import math_utils


class ReferenceMotionState:
    """Encapsulates selected fields from a reference motion."""

    def __init__(
        self,
        ref_motion: dict,
        body_ids: List[int] | None = None,
        joint_ids: List[int] | None = None,
    ):
        """Represents the state of a reference motion frame.

        Args:
            ref_motion (dict): Reference motion data at a specific point in time, loaded from the dataset.
            body_ids (List[int] | None, optional): Desired ordering of the bodies. Defaults to None.
            joint_ids (List[int] | None, optional): Desired ordering of the joints. Defaults to None.
        """

        if body_ids is None:
            num_bodies = ref_motion["rg_pos"].shape[1]
            body_ids = list(range(0, num_bodies))

        if joint_ids is None:
            num_joints = ref_motion["dof_pos"].shape[1]
            joint_ids = list(range(0, num_joints))

        # Root
        self.root_pos = ref_motion["root_pos"]
        self.root_rot = ref_motion["root_rot"]
        self.root_lin_vel = ref_motion["root_vel"]
        self.root_ang_vel = ref_motion["root_ang_vel"]

        # Links
        self.body_pos = ref_motion["rg_pos"][:, body_ids, :]
        self.body_rot = ref_motion["rb_rot"][:, body_ids, :]  # [num_envs, num_markers, 4]
        self.body_lin_vel = ref_motion["body_vel"][:, body_ids, :]  # [num_envs, num_markers, 3]
        self.body_ang_vel = ref_motion["body_ang_vel"][:, body_ids, :]  # [num_envs, num_markers, 3]

        # Extended links (fall back to non-extended if extensions are absent)
        self.body_pos_extend = ref_motion.get("rg_pos_t", ref_motion["rg_pos"][:, body_ids, :])
        self.body_rot_extend = ref_motion.get("rg_rot_t", ref_motion["rb_rot"][:, body_ids, :])
        self.body_lin_vel_extend = ref_motion.get("body_vel_t", ref_motion["body_vel"][:, body_ids, :])
        self.body_ang_vel_extend = ref_motion.get("body_ang_vel_t", ref_motion["body_ang_vel"][:, body_ids, :])

        self.joint_pos = ref_motion["dof_pos"][:, joint_ids]
        self.joint_vel = ref_motion["dof_vel"][:, joint_ids]


@dataclass
class ReferenceMotionManagerCfg:
    motion_path: str = ""
    skeleton_path: str = ""
    fk_frame_rotation: Optional[List[float]] = None
    extend_hand: bool = True
    extend_head: bool = True
    """Optional wxyz quaternion for MJCF/SMPL axis alignment (typically ``[0.5, 0.5, 0.5, 0.5]`` for K1 AMASS retargeting).

    Applies a *similarity transform* to every cached body quaternion, then
    aligns cached positions and world-frame velocities to the new root
    orientation via ``D = R_corr @ R_wrong.T`` about the root position, so
    reference body poses match the simulator after reset.
    """


class ReferenceMotionManager:
    def __init__(
        self,
        cfg: ReferenceMotionManagerCfg,
        device: torch.device,
        num_envs: int,
        random_sample: bool,
        extend_head: bool,
        dt: float,
        extend_hand: bool = True,
    ):
        """Initializes a reference motion manager that loads and queries a motion dataset.

        Args:
            cfg (ReferenceMotionManagerCfg): Configuration that specifies dataset and skeleton paths.
            device (torch.device): Device to host tensors on.
            num_envs (int): Number of environments/instances.
            random_sample (bool): Whether to randomly sample the dataset.
            extend_head (bool): Whether to extend the head of the body for specific robots, e.g. H1.
            dt (float): Length of a policy time step, which is the length of a physics time steps multiplied by decimation.
            extend_hand (bool): Whether to extend virtual hand links. Set False when the robot has physical hand links.
        """
        self._device = device
        self._num_envs = num_envs
        self._dt = dt

        if cfg.fk_frame_rotation is not None:
            q = torch.tensor(cfg.fk_frame_rotation, dtype=torch.float32, device=device)
            q = q / q.norm()
            self._fk_q_R = q
            self._fk_q_R_inv = math_utils.quat_conjugate(q.unsqueeze(0)).squeeze(0)
        else:
            self._fk_q_R = None
            self._fk_q_R_inv = None

        self._motion_lib = MotionLibH1(
            motion_file=cfg.motion_path,
            mjcf_file=cfg.skeleton_path,
            device=self._device,
            masterfoot_conifg=None,
            fix_height=False,
            multi_thread=False,
            extend_head=extend_head,
            extend_hand=extend_hand,
        )

        self._skeleton_trees = [SkeletonTree.from_mjcf(cfg.skeleton_path)] * self._num_envs

        self._motion_ids = torch.arange(self._num_envs).to(self._device)
        self._motion_start_times = torch.zeros(
            self._num_envs, dtype=torch.float32, device=self._device, requires_grad=False
        )

        self.load_motions(random_sample=random_sample, start_idx=0)

    @property
    def motion_lib(self):
        """Returns the MotionLib instance that handles loading and querying the motion dataset."""
        return self._motion_lib

    @property
    def motion_start_times(self):
        """Gets the start times of the loaded motions."""
        return self._motion_start_times

    @property
    def motion_len(self):
        """Gets the lengths of the loaded motions."""
        return self._motion_len

    @property
    def num_unique_motions(self):
        """Returns the number of unique motions in the dataset."""
        return self._motion_lib._num_unique_motions

    @property
    def body_extended_names(self) -> list[str]:
        return self._motion_lib.mesh_parsers.model_names

    def get_motion_num_steps(self):
        """Gets the number of motion steps/frames of the sampled motions."""
        return self._motion_lib.get_motion_num_steps()

    def load_motions(self, random_sample: bool, start_idx: int):
        """Loads motions from the motion dataset."""
        self._motion_lib.load_motions(
            skeleton_trees=self._skeleton_trees,
            gender_betas=[torch.zeros(17)] * self._num_envs,
            limb_weights=[np.zeros(10)] * self._num_envs,
            random_sample=random_sample,
            start_idx=start_idx,
        )
        self._motion_len = self._motion_lib.get_motion_length(self._motion_ids)
        self.reset_motion_start_times(env_ids=self._motion_ids, sample=False)

    def reset_motion_start_times(self, env_ids: torch.Tensor, sample: bool):
        """Resets the time at which the reference motions start playing."""
        if sample:
            self._motion_start_times[env_ids] = self._motion_lib.sample_time(self._motion_ids[env_ids])
        else:
            self._motion_start_times[env_ids] = 0

    def episodes_exceed_motion_length(
        self, episode_times: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Checks if the reference motion has reached to the end."""
        if env_ids is None:
            return (episode_times + self._motion_start_times) > self._motion_len
        return (episode_times[env_ids] + self._motion_start_times[env_ids]) > self._motion_len[env_ids]

    def get_state_from_motion_lib_cache(
        self,
        episode_length_buf: torch.Tensor,
        terrain_heights: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
        quaternion_is_xyzw=True,
    ) -> ReferenceMotionState:
        """Query a reference motion frame from motion lib."""
        motion_times = episode_length_buf * self._dt + self._motion_start_times
        motion_res = self._motion_lib.get_motion_state(self._motion_ids, motion_times, offset=offset)

        if terrain_heights is not None:
            delta_height = terrain_heights.clone()
            if offset is not None:
                delta_height -= offset[:, 2].unsqueeze(1)
            motion_res["root_pos"][:, 2] += delta_height.flatten()
            if "rg_pos" in motion_res:
                motion_res["rg_pos"][:, :, 2] += delta_height
            if "rg_pos_t" in motion_res:
                motion_res["rg_pos_t"][:, :, 2] += delta_height

        # Update quaternion convention
        if quaternion_is_xyzw:
            for key, value in motion_res.items():
                if value.shape[-1] == 4:
                    motion_res[key] = value[..., [3, 0, 1, 2]]

        # Apply FK-frame → Isaac-Lab-Z-up correction when configured.
        # 1) Sandwich all body quaternions (conjugation in the SMPL/MJCF frame).
        # 2) Rigidly re-express cached world-frame quantities about the root so
        #    they match the *new* root orientation: the motion cache was built
        #    with the pre-sandwich root rotation, but the sim is reset with the
        #    post-sandwich root.  Without this, reference body positions drift
        #    from the sim by O(1 m) and ``reference_motion_distance`` fires on
        #    step 1 (episodes ~1–2 steps, joint-velocity tracking rewards ~0).
        if self._fk_q_R is not None:
            qR = self._fk_q_R
            qRi = self._fk_q_R_inv
            root_rot_before = motion_res["root_rot"].clone()
            R_wrong = math_utils.matrix_from_quat(root_rot_before)
            _QUAT_KEYS = {"root_rot", "rb_rot", "rg_rot_t"}
            for key, value in motion_res.items():
                if key in _QUAT_KEYS:
                    orig_shape = value.shape
                    flat = value.reshape(-1, 4)
                    qR_exp = qR.unsqueeze(0).expand_as(flat)
                    qRi_exp = qRi.unsqueeze(0).expand_as(flat)
                    flat = math_utils.quat_mul(qR_exp, math_utils.quat_mul(flat, qRi_exp))
                    motion_res[key] = flat.reshape(orig_shape)

            R_corr = math_utils.matrix_from_quat(motion_res["root_rot"])
            # D maps vectors from the cached FK root frame to the corrected one.
            D = torch.bmm(R_corr, R_wrong.transpose(-1, -2))

            root_p = motion_res["root_pos"].unsqueeze(1)
            for pos_key in ("rg_pos", "rg_pos_t"):
                if pos_key not in motion_res:
                    continue
                diff = motion_res[pos_key] - root_p
                motion_res[pos_key] = root_p + torch.matmul(D.unsqueeze(1), diff.unsqueeze(-1)).squeeze(-1)

            motion_res["root_vel"] = torch.bmm(D, motion_res["root_vel"].unsqueeze(-1)).squeeze(-1)
            for vkey in ("body_vel", "body_vel_t"):
                if vkey not in motion_res:
                    continue
                v = motion_res[vkey]
                motion_res[vkey] = torch.matmul(D.unsqueeze(1), v.unsqueeze(-1)).squeeze(-1)

            motion_res["root_ang_vel"] = torch.bmm(D, motion_res["root_ang_vel"].unsqueeze(-1)).squeeze(-1)
            for wkey in ("body_ang_vel", "body_ang_vel_t"):
                if wkey not in motion_res:
                    continue
                w = motion_res[wkey]
                motion_res[wkey] = torch.matmul(D.unsqueeze(1), w.unsqueeze(-1)).squeeze(-1)

        return ReferenceMotionState(motion_res)
