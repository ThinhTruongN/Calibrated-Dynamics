from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class TriplePendulumEnv(gym.Env):
    def __init__(self, frame_skip=5, max_steps=500):
        model_path = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / "triple_pendulum.xml"
        )

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.current_step = 0

        # Three continuous motor torques between -3 and 3.
        self.action_space = spaces.Box(
            low=self.model.actuator_ctrlrange[:, 0].astype(np.float32),
            high=self.model.actuator_ctrlrange[:, 1].astype(np.float32),
            dtype=np.float32,
        )

        # cos(angle), sin(angle), and velocity for each joint:
        # 3 + 3 + 3 = 9 observed values.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

    def _get_observation(self):
        angles = self.data.qpos.copy()
        angular_velocities = self.data.qvel.copy()

        observation = np.concatenate(
            [
                np.cos(angles),
                np.sin(angles),
                angular_velocities,
            ]
        )

        return observation.astype(np.float32)

    def _calculate_reward(self, action):
        joint_angles = self.data.qpos.copy()

        # Desired configuration:
        # first link upright, remaining links straight.
        target_angles = np.array([
            np.pi,
            0.0,
            0.0,
        ])

        # Shortest periodic distance from each target angle.
        angle_errors = np.arctan2(
            np.sin(joint_angles - target_angles),
            np.cos(joint_angles - target_angles),
        )

        individual_joint_scores = np.clip(
            1.0 - (angle_errors / np.pi) ** 2,
            0.0,
            1.0,
        )

        first_link_score = individual_joint_scores[0]

        # Becomes high only when every joint is correct.
        all_joints_score = np.prod(
            individual_joint_scores
        )

        # The first term provides an easy initial learning signal.
        # The product prevents partial folded poses from scoring highly.
        upright_score = (
            0.4 * first_link_score
            + 0.6 * all_joints_score
        )

        velocity_penalty = 0.001 * np.mean(
            self.data.qvel**2
        )
        torque_penalty = 0.0001 * np.mean(
            action**2
        )

        all_joints_upright = (
            np.min(individual_joint_scores) > 0.98
        )
        moving_slowly = (
            np.linalg.norm(self.data.qvel) < 1.0
        )

        balance_bonus = 0.0

        if all_joints_upright and moving_slowly:
            balance_bonus = 2.0

        reward = (
            2.0 * upright_score
            + balance_bonus
            - velocity_penalty
            - torque_penalty
        )

        return float(reward), float(upright_score)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Begin close to the downward position with slight randomness.
        self.data.qpos[:] = self.np_random.uniform(
            low=-0.05,
            high=0.05,
            size=self.model.nq,
        )

        self.data.qvel[:] = self.np_random.uniform(
            low=-0.05,
            high=0.05,
            size=self.model.nv,
        )

        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0

        return self._get_observation(), {}

    def step(self, action):
        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        self.data.ctrl[:] = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        observation = self._get_observation()
        reward, upright_score = self._calculate_reward(action)

        terminated = not np.all(np.isfinite(observation))
        truncated = self.current_step >= self.max_steps

        info = {
            "upright_score": upright_score,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
