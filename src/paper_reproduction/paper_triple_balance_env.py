from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class PaperTripleBalanceEnv(gym.Env):
    """MuJoCo translation of Manzl et al.'s triple-pendulum balance task."""

    metadata = {"render_modes": []}

    LINK_COUNT = 3
    CONTROL_DT = 0.02
    CART_FORCE = 60.0
    CART_POSITION_THRESHOLD = 5.4
    RELATIVE_ANGLE_THRESHOLD = 3.0 * np.pi / 20.0
    TEST_ERROR_THRESHOLD = 0.75
    REWARD_POSITION_WEIGHT = 0.7
    TRAIN_RESET_RANGE = 0.1
    TEST_RESET_RANGE = 0.05
    DEFAULT_MAX_STEPS = 2048

    def __init__(self, max_steps=DEFAULT_MAX_STEPS, reset_range=TRAIN_RESET_RANGE):
        super().__init__()

        model_path = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / "paper_triple_balance.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        if (self.model.nq, self.model.nv, self.model.nu) != (4, 4, 1):
            raise ValueError(
                "Expected one slide, three hinges, and one actuator; "
                f"got nq={self.model.nq}, nv={self.model.nv}, "
                f"nu={self.model.nu}."
            )
        if not np.isclose(self.model.opt.timestep, self.CONTROL_DT):
            raise ValueError("The MuJoCo step must equal the paper's 20 ms control step.")

        self.max_steps = int(max_steps)
        self.reset_range = float(reset_range)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.reset_range < 0.0:
            raise ValueError("reset_range cannot be negative")

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

        actuator_joint_ids = self.model.actuator_trnid[:, 0]
        if actuator_joint_ids.tolist() != [self._cart_joint_id]:
            raise ValueError("The only actuator must drive cart_slide.")

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
            raise ValueError("All link endpoint sites must exist.")
        self._tip_site_id = int(self._endpoint_site_ids[-1])

        # The paper uses two bang-bang actions, not a continuous action.
        self.action_space = spaces.Discrete(2)

        position_high = np.array(
            [self.CART_POSITION_THRESHOLD * 2.0]
            + [self.RELATIVE_ANGLE_THRESHOLD * 2.0] * self.LINK_COUNT,
            dtype=np.float32,
        )
        velocity_high = np.full(4, np.finfo(np.float32).max, dtype=np.float32)
        high = np.concatenate([position_high, velocity_high])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _joint_id(self, name):
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"Missing joint in model: {name}")
        return joint_id

    def _position_state(self):
        return np.concatenate(
            [
                [self.data.qpos[self._cart_qpos_index]],
                self.data.qpos[self._hinge_qpos_indices],
            ]
        )

    def _velocity_state(self):
        return np.concatenate(
            [
                [self.data.qvel[self._cart_qvel_index]],
                self.data.qvel[self._hinge_qvel_indices],
            ]
        )

    def _get_observation(self):
        return np.concatenate(
            [self._position_state(), self._velocity_state()]
        ).astype(np.float32)

    def _set_state(self, position, velocity=None):
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (4,):
            raise ValueError("position must contain cart x and three relative angles")
        if velocity is None:
            velocity = np.zeros(4, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        if velocity.shape != (4,):
            raise ValueError("velocity must contain cart and three angular velocities")

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self._cart_qpos_index] = position[0]
        self.data.qpos[self._hinge_qpos_indices] = position[1:]
        self.data.qvel[self._cart_qvel_index] = velocity[0]
        self.data.qvel[self._hinge_qvel_indices] = velocity[1:]
        mujoco.mj_forward(self.model, self.data)

    def _tip_x(self):
        return float(self.data.site_xpos[self._tip_site_id, 0])

    def _failed(self):
        position = self._position_state()
        return bool(
            not np.all(np.isfinite(self._get_observation()))
            or abs(position[0]) > self.CART_POSITION_THRESHOLD
            or np.any(np.abs(position[1:]) > self.RELATIVE_ANGLE_THRESHOLD)
        )

    def _calculate_reward(self):
        last_relative_angle = float(self.data.qpos[self._hinge_qpos_indices[-1]])
        tip_x = self._tip_x()
        position_term = (
            self.REWARD_POSITION_WEIGHT
            * abs(tip_x)
            / self.CART_POSITION_THRESHOLD
        )
        angle_term = (
            (1.0 - self.REWARD_POSITION_WEIGHT)
            * abs(last_relative_angle)
            / self.RELATIVE_ANGLE_THRESHOLD
        )
        return float(1.0 - position_term - angle_term)

    def _build_info(self, force):
        position = self._position_state()
        return {
            "cart_force": float(force),
            "tip_x": self._tip_x(),
            "cart_position": float(position[0]),
            "relative_angles": position[1:].astype(np.float32),
            "position_error_inf": float(np.max(np.abs(position))),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        reset_range = self.reset_range
        if options is not None and "reset_range" in options:
            reset_range = float(options["reset_range"])
            if reset_range < 0.0:
                raise ValueError("reset_range cannot be negative")

        state = self.np_random.uniform(-reset_range, reset_range, 8)
        self._set_state(state[:4], state[4:])
        self.current_step = 0
        return self._get_observation(), self._build_info(force=0.0)

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        force = self.CART_FORCE if action == 1 else -self.CART_FORCE
        self.data.ctrl[0] = force
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._failed()
        truncated = self.current_step >= self.max_steps
        return observation, reward, terminated, truncated, self._build_info(force)

    def close(self):
        pass
