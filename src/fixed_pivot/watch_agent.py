from pathlib import Path
import time

import mujoco.viewer
from stable_baselines3 import SAC

from triple_pendulum_env import TriplePendulumEnv


project_root = Path(__file__).resolve().parents[2]
model_path = (
    project_root
    / "checkpoints"
    / "sac_triple_pendulum_v3_100k.zip"
)

env = TriplePendulumEnv()
model = SAC.load(str(model_path))

observation, info = env.reset(seed=42)

episode_number = 1
episode_reward = 0.0
maximum_upright_score = 0.0

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running() and episode_number <= 3:
        step_start = time.time()

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        episode_reward += reward
        maximum_upright_score = max(
            maximum_upright_score,
            info["upright_score"],
        )

        viewer.sync()

        if terminated or truncated:
            print(
                f"Episode {episode_number}: "
                f"reward={episode_reward:.2f}, "
                f"maximum upright score="
                f"{maximum_upright_score:.3f}"
            )

            episode_number += 1
            episode_reward = 0.0
            maximum_upright_score = 0.0
            observation, info = env.reset()

        simulation_step_time = (
            env.model.opt.timestep * env.frame_skip
        )
        remaining_time = simulation_step_time - (
            time.time() - step_start
        )

        if remaining_time > 0:
            time.sleep(remaining_time)

env.close()
