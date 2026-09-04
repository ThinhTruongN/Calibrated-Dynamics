import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
from stable_baselines3 import SAC

from cart_triple_pendulum_env import CartTriplePendulumEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch and evaluate a cart-triple SAC policy."
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def newest_cart_model(project_root):
    candidates = sorted(
        (project_root / "checkpoints").glob(
            "cart_triple_*/sac_cart_triple_pendulum_*_steps.zip"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No final cart-triple model found. Train one or pass --model."
        )
    return candidates[0]


def empty_result():
    return {
        "reward": 0.0,
        "peak_min_link_score": 0.0,
        "peak_endpoint_height_score": 0.0,
        "max_consecutive_balanced_steps": 0,
        "success": False,
    }


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    project_root = Path(__file__).resolve().parents[2]
    model_path = (args.model or newest_cart_model(project_root)).resolve()
    env = CartTriplePendulumEnv()
    model = SAC.load(str(model_path), device="cpu")
    observation, _ = env.reset(seed=args.seed)
    results = []
    current = empty_result()

    print(f"Loaded: {model_path}")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        camera_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_CAMERA, "side"
        )
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id

        while viewer.is_running() and len(results) < args.episodes:
            step_start = time.perf_counter()
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            current["reward"] += reward
            current["peak_min_link_score"] = max(
                current["peak_min_link_score"], info["min_individual_link_score"]
            )
            current["peak_endpoint_height_score"] = max(
                current["peak_endpoint_height_score"], info["endpoint_height_score"]
            )
            current["max_consecutive_balanced_steps"] = max(
                current["max_consecutive_balanced_steps"],
                info["consecutive_balanced_steps"],
            )
            current["success"] = current["success"] or info["success"]
            viewer.sync()

            if terminated or truncated:
                results.append(current)
                episode_number = len(results)
                print(
                    f"Episode {episode_number}: reward={current['reward']:.2f}, "
                    f"success={current['success']}, "
                    f"peak min-link={current['peak_min_link_score']:.3f}, "
                    f"peak endpoint-height={current['peak_endpoint_height_score']:.3f}, "
                    f"max balanced steps={current['max_consecutive_balanced_steps']}"
                )
                if episode_number < args.episodes:
                    observation, _ = env.reset(seed=args.seed + episode_number)
                    current = empty_result()

            remaining = env.model.opt.timestep * env.frame_skip - (
                time.perf_counter() - step_start
            )
            if remaining > 0.0:
                time.sleep(remaining)

    env.close()
    if results:
        successes = sum(result["success"] for result in results)
        print(f"Success rate: {successes / len(results):.1%} "
              f"({successes}/{len(results)})")


if __name__ == "__main__":
    main()
