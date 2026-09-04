import argparse
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import SAC

from underactuated_triple_pendulum_env import UnderactuatedTriplePendulumEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch and strictly evaluate an underactuated SAC policy."
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Model .zip path. Defaults to the newest final underactuated model.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def newest_underactuated_model(project_root):
    candidates = sorted(
        (project_root / "checkpoints").glob(
            "underactuated_*/sac_underactuated_triple_pendulum_*_steps.zip"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No final underactuated model found. Pass one with --model after training."
        )
    return candidates[0]


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    project_root = Path(__file__).resolve().parents[2]
    model_path = args.model or newest_underactuated_model(project_root)
    model_path = model_path.resolve()

    env = UnderactuatedTriplePendulumEnv()
    model = SAC.load(str(model_path), device="cpu")
    observation, _ = env.reset(seed=args.seed)
    episode_results = []
    current_result = {
        "reward": 0.0,
        "peak_min_link_score": 0.0,
        "peak_endpoint_height_score": 0.0,
        "max_consecutive_balanced_steps": 0,
        "success": False,
    }

    print(f"Loaded: {model_path}")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and len(episode_results) < args.episodes:
            step_start = time.perf_counter()
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)

            current_result["reward"] += reward
            current_result["peak_min_link_score"] = max(
                current_result["peak_min_link_score"],
                info["min_individual_link_score"],
            )
            current_result["peak_endpoint_height_score"] = max(
                current_result["peak_endpoint_height_score"],
                info["endpoint_height_score"],
            )
            current_result["max_consecutive_balanced_steps"] = max(
                current_result["max_consecutive_balanced_steps"],
                info["consecutive_balanced_steps"],
            )
            current_result["success"] = current_result["success"] or info["success"]
            viewer.sync()

            if terminated or truncated:
                episode_results.append(current_result)
                episode_number = len(episode_results)
                print(
                    f"Episode {episode_number}: "
                    f"reward={current_result['reward']:.2f}, "
                    f"success={current_result['success']}, "
                    f"peak min-link={current_result['peak_min_link_score']:.3f}, "
                    "peak endpoint-height="
                    f"{current_result['peak_endpoint_height_score']:.3f}, "
                    "max balanced steps="
                    f"{current_result['max_consecutive_balanced_steps']}"
                )
                if episode_number < args.episodes:
                    observation, _ = env.reset(seed=args.seed + episode_number)
                    current_result = {
                        "reward": 0.0,
                        "peak_min_link_score": 0.0,
                        "peak_endpoint_height_score": 0.0,
                        "max_consecutive_balanced_steps": 0,
                        "success": False,
                    }

            simulation_step_time = env.model.opt.timestep * env.frame_skip
            remaining = simulation_step_time - (time.perf_counter() - step_start)
            if remaining > 0.0:
                time.sleep(remaining)

    env.close()
    if episode_results:
        successes = sum(result["success"] for result in episode_results)
        print(
            f"Success rate: {successes / len(episode_results):.1%} "
            f"({successes}/{len(episode_results)})"
        )


if __name__ == "__main__":
    main()
