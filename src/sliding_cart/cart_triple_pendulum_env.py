from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class CartTriplePendulumEnv(gym.Env):
    """Three passive pendulum hinges mounted on an actuated horizontal cart."""

    metadata = {"render_modes": []}

    LINK_COUNT = 3
    UPRIGHT_HINGE_QPOS = np.array([np.pi, 0.0, 0.0], dtype=np.float64)
    BALANCE_ANGLE_TOLERANCE = np.deg2rad(12.0)
    BALANCE_ANGULAR_VELOCITY_TOLERANCE = 0.5
    BALANCE_CART_VELOCITY_TOLERANCE = 0.35
    BALANCE_CART_POSITION_TOLERANCE = 0.75
    BALANCE_HEIGHT_THRESHOLD = 0.97
    SUCCESS_BALANCED_STEPS = 50

    def __init__(self, frame_skip=5, max_steps=1000):
        super().__init__()

        model_path = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / "cart_triple_pendulum.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        if (self.model.nq, self.model.nv, self.model.nu) != (4, 4, 1):
            raise ValueError(
                "Expected one slide, three hinges, and one actuator; "
                f"got nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}."
            )

        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.current_step = 0
        self.consecutive_balanced_steps = 0

        self._cart_joint_id = self._joint_id("cart_slide")
        self._hinge_joint_ids = np.array(
            [self._joint_id(f"joint{index}") for index in range(1, 4)],
            dtype=np.int32,
        )
        self._cart_qpos_index = int(self.model.jnt_qposadr[self._cart_joint_id])
        self._cart_qvel_index = int(self.model.jnt_dofadr[self._cart_joint_id])
        self._hinge_qpos_indices = self.model.jnt_qposadr[
            self._hinge_joint_ids
        ].astype(np.int32)
        self._hinge_qvel_indices = self.model.jnt_dofadr[
            self._hinge_joint_ids
        ].astype(np.int32)
        self._cart_range = self.model.jnt_range[self._cart_joint_id].copy()
        self._cart_position_scale = float(np.max(np.abs(self._cart_range)))

        actuator_joint_ids = self.model.actuator_trnid[:, 0]
        if actuator_joint_ids.tolist() != [self._cart_joint_id]:
            raise ValueError("The sole actuator must drive only cart_slide.")

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
            np.zeros(self.LINK_COUNT, dtype=np.float64)
        )
        self._up_endpoint_heights = self._endpoint_heights_at(
            self.UPRIGHT_HINGE_QPOS
        )
        self._endpoint_height_ranges = (
            self._up_endpoint_heights - self._down_endpoint_heights
        )

        self.action_space = spaces.Box(
            low=self.model.actuator_ctrlrange[:, 0].astype(np.float32),
            high=self.model.actuator_ctrlrange[:, 1].astype(np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        # cos/sin hinges, hinge velocities, cart x/vx, endpoint heights: 14.
        self.observation_space = spaces.Box(
            low=np.concatenate(
                [
                    -np.ones(6, dtype=np.float32),
                    np.full(3, -np.inf, dtype=np.float32),
                    np.array([-1.0, -np.inf], dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                ]
            ),
            high=np.concatenate(
                [
                    np.ones(6, dtype=np.float32),
                    np.full(3, np.inf, dtype=np.float32),
                    np.array([1.0, np.inf], dtype=np.float32),
                    np.ones(3, dtype=np.float32),
                ]
            ),
            dtype=np.float32,
        )

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _joint_id(self, name):
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"Missing joint in model: {name}")
        return joint_id

    def _set_configuration(self, cart_position, hinge_qpos):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self._cart_qpos_index] = cart_position
        self.data.qpos[self._hinge_qpos_indices] = hinge_qpos
        mujoco.mj_forward(self.model, self.data)

    def _endpoint_heights_at(self, hinge_qpos):
        self._set_configuration(0.0, hinge_qpos)
        return self.data.site_xpos[self._endpoint_site_ids, 2].copy()

    def _normalized_endpoint_heights(self):
        heights = self.data.site_xpos[self._endpoint_site_ids, 2]
        normalized = (heights - self._down_endpoint_heights) / (
            self._endpoint_height_ranges
        )
        return np.clip(normalized, 0.0, 1.0)

    def _global_link_angle_errors(self):
        hinge_qpos = self.data.qpos[self._hinge_qpos_indices]
        global_angles = np.cumsum(hinge_qpos)
        return np.arctan2(
            np.sin(global_angles - np.pi),
            np.cos(global_angles - np.pi),
        )

    def _state_metrics(self):
        angle_errors = self._global_link_angle_errors()
        individual_link_scores = 0.5 * (1.0 + np.cos(angle_errors))
        min_individual_link_score = float(np.min(individual_link_scores))
        all_link_score = float(np.cbrt(np.prod(individual_link_scores)))
        endpoint_height_score = float(np.min(self._normalized_endpoint_heights()))

        hinge_qvel = self.data.qvel[self._hinge_qvel_indices]
        max_angular_velocity = float(np.max(np.abs(hinge_qvel)))
        cart_position = float(self.data.qpos[self._cart_qpos_index])
        cart_velocity = float(self.data.qvel[self._cart_qvel_index])

        all_links_close = bool(
            np.all(np.abs(angle_errors) < self.BALANCE_ANGLE_TOLERANCE)
        )
        is_balanced = bool(
            all_links_close
            and endpoint_height_score > self.BALANCE_HEIGHT_THRESHOLD
            and max_angular_velocity < self.BALANCE_ANGULAR_VELOCITY_TOLERANCE
            and abs(cart_velocity) < self.BALANCE_CART_VELOCITY_TOLERANCE
            and abs(cart_position) < self.BALANCE_CART_POSITION_TOLERANCE
        )

        return {
            "individual_link_scores": individual_link_scores,
            "min_individual_link_score": min_individual_link_score,
            "all_link_score": all_link_score,
            "endpoint_height_score": endpoint_height_score,
            "max_angular_velocity": max_angular_velocity,
            "cart_position": cart_position,
            "cart_velocity": cart_velocity,
            "is_balanced": is_balanced,
        }

    def _get_observation(self):
        hinge_qpos = self.data.qpos[self._hinge_qpos_indices]
        hinge_qvel = self.data.qvel[self._hinge_qvel_indices]
        cart_position = self.data.qpos[self._cart_qpos_index]
        cart_velocity = self.data.qvel[self._cart_qvel_index]
        normalized_cart_position = np.clip(
            cart_position / self._cart_position_scale, -1.0, 1.0
        )
        observation = np.concatenate(
            [
                np.cos(hinge_qpos),
                np.sin(hinge_qpos),
                hinge_qvel,
                [normalized_cart_position, cart_velocity],
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
        hinge_qvel = self.data.qvel[self._hinge_qvel_indices]
        angular_stillness = float(np.exp(-np.mean(hinge_qvel**2)))
        cart_stillness = float(np.exp(-(metrics["cart_velocity"] / 0.5) ** 2))
        centered_score = float(np.exp(-(metrics["cart_position"] / 1.0) ** 2))
        normalized_action = action / self.action_space.high

        reward_terms = {
            "mean_link_reward": 0.5 * mean_link_score,
            "all_link_reward": 3.0 * all_link_score,
            "coordinated_height_reward": 2.0 * coordinated_height_score,
            "balanced_stillness_reward": (
                1.5 * all_link_score * angular_stillness * cart_stillness
            ),
            "centered_cart_reward": 0.5 * all_link_score * centered_score,
            "balance_bonus": 10.0 if metrics["is_balanced"] else 0.0,
            "force_penalty": 0.01 * float(np.mean(normalized_action**2)),
            "track_edge_penalty": 0.2
            * (abs(metrics["cart_position"]) / self._cart_position_scale) ** 4,
        }
        reward = (
            reward_terms["mean_link_reward"]
            + reward_terms["all_link_reward"]
            + reward_terms["coordinated_height_reward"]
            + reward_terms["balanced_stillness_reward"]
            + reward_terms["centered_cart_reward"]
            + reward_terms["balance_bonus"]
            - reward_terms["force_penalty"]
            - reward_terms["track_edge_penalty"]
        )
        return float(reward), reward_terms

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._cart_qpos_index] = self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[self._hinge_qpos_indices] = self.np_random.uniform(
            -0.05, 0.05, self.LINK_COUNT
        )
        self.data.qvel[:] = self.np_random.uniform(-0.05, 0.05, self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self.consecutive_balanced_steps = 0
        metrics = self._state_metrics()
        return self._get_observation(), self._build_info(metrics, {})

    def _build_info(self, metrics, reward_terms):
        return {
            "individual_link_scores": metrics["individual_link_scores"].astype(
                np.float32
            ),
            "min_individual_link_score": metrics["min_individual_link_score"],
            "endpoint_height_score": metrics["endpoint_height_score"],
            "max_angular_velocity": metrics["max_angular_velocity"],
            "cart_position": metrics["cart_position"],
            "cart_velocity": metrics["cart_velocity"],
            "is_balanced": metrics["is_balanced"],
            "consecutive_balanced_steps": self.consecutive_balanced_steps,
            "success": self.consecutive_balanced_steps >= self.SUCCESS_BALANCED_STEPS,
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
        if terminated:
            reward = -100.0

        return (
            observation,
            reward,
            terminated,
            truncated,
            self._build_info(metrics, reward_terms),
        )

    def close(self):
        pass
