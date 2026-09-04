import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from underactuated_triple_pendulum_env import UnderactuatedTriplePendulumEnv


def evaluate_strict_metrics(model, episodes=10, seed=10_000):
    if episodes <= 0:
        raise ValueError("Evaluation episodes must be positive")

    env = UnderactuatedTriplePendulumEnv()
    results = []

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_reward = 0.0
            peak_min_link_score = 0.0
            peak_endpoint_height_score = 0.0
            max_consecutive_balanced_steps = 0
            succeeded = False

            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                peak_min_link_score = max(
                    peak_min_link_score, info["min_individual_link_score"]
                )
                peak_endpoint_height_score = max(
                    peak_endpoint_height_score, info["endpoint_height_score"]
                )
                max_consecutive_balanced_steps = max(
                    max_consecutive_balanced_steps,
                    info["consecutive_balanced_steps"],
                )
                succeeded = succeeded or info["success"]

                if terminated or truncated:
                    break

            results.append(
                {
                    "reward": episode_reward,
                    "peak_min_link_score": peak_min_link_score,
                    "peak_endpoint_height_score": peak_endpoint_height_score,
                    "max_consecutive_balanced_steps": max_consecutive_balanced_steps,
                    "success": succeeded,
                }
            )
    finally:
        env.close()

    success_rate = sum(result["success"] for result in results) / episodes
    mean_reward = sum(result["reward"] for result in results) / episodes
    mean_peak_min_link = (
        sum(result["peak_min_link_score"] for result in results) / episodes
    )
    mean_peak_height = (
        sum(result["peak_endpoint_height_score"] for result in results) / episodes
    )
    mean_max_consecutive = (
        sum(result["max_consecutive_balanced_steps"] for result in results)
        / episodes
    )

    print("Strict deterministic evaluation:")
    print(f"  success rate: {success_rate:.1%} ({sum(r['success'] for r in results)}/{episodes})")
    print(f"  mean episode reward: {mean_reward:.2f}")
    print(f"  mean peak minimum individual-link score: {mean_peak_min_link:.3f}")
    print(f"  mean peak endpoint-height score: {mean_peak_height:.3f}")
    print(f"  mean maximum consecutive balanced steps: {mean_max_consecutive:.1f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CPU SAC on the underactuated triple pendulum."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        required=True,
        help="Number of environment steps to train (must be specified explicitly).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be positive")

    project_root = Path(__file__).resolve().parents[2]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = project_root / "checkpoints" / f"underactuated_{run_id}"
    run_directory.mkdir(parents=True, exist_ok=False)

    env = Monitor(UnderactuatedTriplePendulumEnv())
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(run_directory),
        name_prefix="sac_underactuated_triple",
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        device="cpu",
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
        print(f"Starting underactuated SAC training for {args.timesteps:,} steps")
        print(f"Checkpoints will be written only to: {run_directory}")
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

        model_path = run_directory / (
            f"sac_underactuated_triple_pendulum_{run_id}_{args.timesteps}_steps"
        )
        model.save(str(model_path))
        print(f"Final model saved to: {model_path}.zip")
        evaluate_strict_metrics(model, episodes=args.eval_episodes)
    finally:
        env.close()


if __name__ == "__main__":
    main()
