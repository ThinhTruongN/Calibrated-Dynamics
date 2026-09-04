import argparse
import json
from datetime import datetime
from pathlib import Path

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from paper_triple_balance_env import PaperTripleBalanceEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the paper-style triple-pendulum balance controller."
    )
    parser.add_argument("--algorithm", choices=("a2c", "ppo"), default="a2c")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    return parser.parse_args()


def build_model(algorithm, env, seed):
    common = {
        "policy": "MlpPolicy",
        "env": env,
        "seed": seed,
        "device": "cpu",
        "verbose": 1,
        "policy_kwargs": {"net_arch": [64, 64]},
    }
    if algorithm == "a2c":
        return A2C(learning_rate=7e-4, **common)
    return PPO(learning_rate=5e-4, n_steps=2048, **common)


def main():
    args = parse_args()
    if args.timesteps <= 0 or args.checkpoint_freq <= 0:
        raise ValueError("timesteps and checkpoint-freq must be positive")

    project_root = Path(__file__).resolve().parents[2]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = (
        project_root
        / "checkpoints"
        / f"paper_triple_balance_{args.algorithm}_{run_id}"
    )
    run_directory.mkdir(parents=True, exist_ok=False)

    configuration = {
        "paper": "Manzl et al., DOI 10.1007/s11044-023-09960-2",
        "algorithm": args.algorithm,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "control_dt": PaperTripleBalanceEnv.CONTROL_DT,
        "cart_force": PaperTripleBalanceEnv.CART_FORCE,
        "train_reset_range": PaperTripleBalanceEnv.TRAIN_RESET_RANGE,
        "reward_position_weight": PaperTripleBalanceEnv.REWARD_POSITION_WEIGHT,
    }
    with (run_directory / "configuration.json").open("w", encoding="ascii") as file:
        json.dump(configuration, file, indent=2)

    env = Monitor(PaperTripleBalanceEnv())
    callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(run_directory),
        name_prefix=f"{args.algorithm}_paper_triple_balance",
    )
    model = build_model(args.algorithm, env, args.seed)

    try:
        print(
            f"Starting paper-style {args.algorithm.upper()} training for "
            f"{args.timesteps:,} steps on CPU"
        )
        print(f"New checkpoints will be written only to: {run_directory}")
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model_path = run_directory / (
            f"{args.algorithm}_paper_triple_balance_{args.timesteps}_steps"
        )
        model.save(str(model_path))
        print(f"Final model saved to: {model_path}.zip")
        print("Run evaluate_paper_triple_balance.py for the 100-test protocol.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
