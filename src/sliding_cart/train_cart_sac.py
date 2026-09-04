import argparse
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from cart_triple_pendulum_env import CartTriplePendulumEnv


def evaluate_strict_metrics(model, episodes=10, seed=10_000):
    if episodes <= 0:
        raise ValueError("Evaluation episodes must be positive")

    env = CartTriplePendulumEnv()
    results = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            result = {
                "reward": 0.0,
                "peak_min_link_score": 0.0,
                "peak_endpoint_height_score": 0.0,
                "max_consecutive_balanced_steps": 0,
                "success": False,
            }

            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                result["reward"] += reward
                result["peak_min_link_score"] = max(
                    result["peak_min_link_score"],
                    info["min_individual_link_score"],
                )
                result["peak_endpoint_height_score"] = max(
                    result["peak_endpoint_height_score"],
                    info["endpoint_height_score"],
                )
                result["max_consecutive_balanced_steps"] = max(
                    result["max_consecutive_balanced_steps"],
                    info["consecutive_balanced_steps"],
                )
                result["success"] = result["success"] or info["success"]
                if terminated or truncated:
                    break

            results.append(result)
    finally:
        env.close()

    successes = sum(result["success"] for result in results)
    print("Strict deterministic evaluation:")
    print(f"  success rate: {successes / episodes:.1%} ({successes}/{episodes})")
    print(f"  mean episode reward: "
          f"{sum(r['reward'] for r in results) / episodes:.2f}")
    print("  mean peak minimum individual-link score: "
          f"{sum(r['peak_min_link_score'] for r in results) / episodes:.3f}")
    print("  mean peak endpoint-height score: "
          f"{sum(r['peak_endpoint_height_score'] for r in results) / episodes:.3f}")
    print("  mean maximum consecutive balanced steps: "
          f"{sum(r['max_consecutive_balanced_steps'] for r in results) / episodes:.1f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAC on the cart-mounted triple pendulum."
    )
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cpu",
        help="Neural-network device. MuJoCo physics always runs on the CPU.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.timesteps <= 0 or args.eval_episodes <= 0:
        raise ValueError("Timesteps and evaluation episodes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but CUDA is unavailable in PyTorch."
        )

    project_root = Path(__file__).resolve().parents[2]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = project_root / "checkpoints" / f"cart_triple_{run_id}"
    run_directory.mkdir(parents=True, exist_ok=False)

    env = Monitor(CartTriplePendulumEnv())
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(run_directory),
        name_prefix="sac_cart_triple",
    )
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        device=args.device,
        learning_rate=3e-4,
        learning_starts=10_000,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs={"net_arch": [256, 256]},
    )

    try:
        print(f"Starting cart-triple SAC training for {args.timesteps:,} steps")
        print(f"Checkpoints will be written only to: {run_directory}")
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
        model_path = run_directory / (
            f"sac_cart_triple_pendulum_{run_id}_{args.timesteps}_steps"
        )
        model.save(str(model_path))
        print(f"Final model saved to: {model_path}.zip")
        evaluate_strict_metrics(model, episodes=args.eval_episodes)
    finally:
        env.close()


if __name__ == "__main__":
    main()
