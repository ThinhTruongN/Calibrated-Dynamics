from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class UnderactuatedTriplePendulumEnv(gym.Env):
    """Triple pendulum with torque only at the fixed base hinge.

    MuJoCo hinge coordinates are relative to each body's parent. With the XML
    geometry extending along local -z, qpos=[0, 0, 0] hangs straight down and
    qpos=[pi, 0, 0] is straight up. Global link angles are therefore the
    cumulative sums of the three relative hinge coordinates.
    """

    metadata = {"render_modes": []}

    LINK_COUNT = 3
    UPRIGHT_QPOS = np.array([np.pi, 0.0, 0.0], dtype=np.float64)
    BALANCE_ANGLE_TOLERANCE = np.deg2rad(12.0)
    BALANCE_VELOCITY_TOLERANCE = 0.5
    BALANCE_HEIGHT_THRESHOLD = 0.97
    SUCCESS_BALANCED_STEPS = 50

    def __init__(self, frame_skip=5, max_steps=1000):
        super().__init__()

        model_path = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / "underactuated_triple_pendulum.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        if (self.model.nq, self.model.nv, self.model.nu) != (3, 3, 1):
            raise ValueError(
                "Expected three hinge coordinates and exactly one actuator; "
                f"got nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}."
            )

        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.current_step = 0
        self.consecutive_balanced_steps = 0

        self._endpoint_site_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    f"link{index}_endpoint",
                )
                for index in range(1, self.LINK_COUNT + 1)
            ],
            dtype=np.int32,
        )
        if np.any(self._endpoint_site_ids < 0):
            raise ValueError("All three named endpoint sites must exist in the XML.")

        self._down_endpoint_heights = self._endpoint_heights_at(
            np.zeros(self.model.nq, dtype=np.float64)
        )
        self._up_endpoint_heights = self._endpoint_heights_at(self.UPRIGHT_QPOS)
        self._endpoint_height_ranges = (
            self._up_endpoint_heights - self._down_endpoint_heights
        )
        if np.any(self._endpoint_height_ranges <= 0.0):
            raise ValueError("Upright endpoint heights must exceed hanging heights.")

        self.action_space = spaces.Box(
            low=self.model.actuator_ctrlrange[:, 0].astype(np.float32),
            high=self.model.actuator_ctrlrange[:, 1].astype(np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        # cos(q), sin(q), qvel, normalized endpoint heights: 3 + 3 + 3 + 3.
        self.observation_space = spaces.Box(
            low=np.concatenate(
                [
                    -np.ones(6, dtype=np.float32),
                    np.full(3, -np.inf, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                ]
            ),
            high=np.concatenate(
                [
                    np.ones(6, dtype=np.float32),
                    np.full(3, np.inf, dtype=np.float32),
                    np.ones(3, dtype=np.float32),
                ]
            ),
            dtype=np.float32,
        )

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _endpoint_heights_at(self, qpos):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self._endpoint_site_ids, 2].copy()

    def _normalized_endpoint_heights(self):
        heights = self.data.site_xpos[self._endpoint_site_ids, 2]
        normalized = (heights - self._down_endpoint_heights) / (
            self._endpoint_height_ranges
        )
        return np.clip(normalized, 0.0, 1.0)

    def _global_link_angle_errors(self):
        global_angles = np.cumsum(self.data.qpos)
        return np.arctan2(
            np.sin(global_angles - np.pi),
            np.cos(global_angles - np.pi),
        )

    def _state_metrics(self):
        angle_errors = self._global_link_angle_errors()
        individual_link_scores = 0.5 * (1.0 + np.cos(angle_errors))
        min_individual_link_score = float(np.min(individual_link_scores))
        all_link_score = float(np.cbrt(np.prod(individual_link_scores)))

        normalized_heights = self._normalized_endpoint_heights()
        endpoint_height_score = float(np.min(normalized_heights))
        max_angular_velocity = float(np.max(np.abs(self.data.qvel)))

        all_links_close = bool(
            np.all(np.abs(angle_errors) < self.BALANCE_ANGLE_TOLERANCE)
        )
        all_endpoints_high = endpoint_height_score > self.BALANCE_HEIGHT_THRESHOLD
        moving_slowly = max_angular_velocity < self.BALANCE_VELOCITY_TOLERANCE
        is_balanced = bool(all_links_close and all_endpoints_high and moving_slowly)

        return {
            "angle_errors": angle_errors,
            "individual_link_scores": individual_link_scores,
            "min_individual_link_score": min_individual_link_score,
            "all_link_score": all_link_score,
            "endpoint_height_score": endpoint_height_score,
            "normalized_endpoint_heights": normalized_heights,
            "max_angular_velocity": max_angular_velocity,
            "is_balanced": is_balanced,
        }

    def _get_observation(self):
        relative_angles = self.data.qpos.copy()
        observation = np.concatenate(
            [
                np.cos(relative_angles),
                np.sin(relative_angles),
                self.data.qvel.copy(),
                self._normalized_endpoint_heights(),
            ]
        )
        return observation.astype(np.float32)

    def _calculate_reward(self, action, metrics):
        individual_scores = metrics["individual_link_scores"]
        min_link_score = metrics["min_individual_link_score"]
        all_link_score = metrics["all_link_score"]
        endpoint_height_score = metrics["endpoint_height_score"]

        mean_link_score = float(np.mean(individual_scores))
        coordinated_height_score = min_link_score * endpoint_height_score
        stillness_score = float(np.exp(-np.mean(self.data.qvel**2)))
        balanced_stillness_score = all_link_score * stillness_score

        normalized_action = action / self.action_space.high
        torque_penalty = 0.01 * float(np.mean(normalized_action**2))
        balance_bonus = 10.0 if metrics["is_balanced"] else 0.0

        reward_terms = {
            "mean_link_reward": 0.5 * mean_link_score,
            "all_link_reward": 3.0 * all_link_score,
            "coordinated_height_reward": 2.0 * coordinated_height_score,
            "balanced_stillness_reward": 1.5 * balanced_stillness_score,
            "balance_bonus": balance_bonus,
            "torque_penalty": torque_penalty,
        }
        reward = (
            reward_terms["mean_link_reward"]
            + reward_terms["all_link_reward"]
            + reward_terms["coordinated_height_reward"]
            + reward_terms["balanced_stillness_reward"]
            + reward_terms["balance_bonus"]
            - reward_terms["torque_penalty"]
        )
        return float(reward), reward_terms

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # qpos=0 is the straight-down state in this model.
        self.data.qpos[:] = self.np_random.uniform(-0.05, 0.05, self.model.nq)
        self.data.qvel[:] = self.np_random.uniform(-0.05, 0.05, self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self.consecutive_balanced_steps = 0
        metrics = self._state_metrics()
        info = self._build_info(metrics, {})
        return self._get_observation(), info

    def _build_info(self, metrics, reward_terms):
        return {
            "individual_link_scores": metrics["individual_link_scores"].astype(
                np.float32
            ),
            "min_individual_link_score": metrics["min_individual_link_score"],
            "endpoint_height_score": metrics["endpoint_height_score"],
            "normalized_endpoint_heights": metrics[
                "normalized_endpoint_heights"
            ].astype(np.float32),
            "max_angular_velocity": metrics["max_angular_velocity"],
            "is_balanced": metrics["is_balanced"],
            "consecutive_balanced_steps": self.consecutive_balanced_steps,
            "success": (
                self.consecutive_balanced_steps >= self.SUCCESS_BALANCED_STEPS
            ),
            "reward_terms": reward_terms,
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1
        observation = self._get_observation()
        metrics = self._state_metrics()

        if metrics["is_balanced"]:
            self.consecutive_balanced_steps += 1
        else:
            self.consecutive_balanced_steps = 0

        reward, reward_terms = self._calculate_reward(action, metrics)
        terminated = not np.all(np.isfinite(observation))
        truncated = self.current_step >= self.max_steps
        info = self._build_info(metrics, reward_terms)

        if terminated:
            reward = -100.0

        return observation, reward, terminated, truncated, info

    def close(self):
        pass
