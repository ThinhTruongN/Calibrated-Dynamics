import argparse
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import A2C, PPO

from paper_triple_balance_env import PaperTripleBalanceEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch a trained paper-style triple balance controller."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("a2c", "ppo"), default="a2c")
    parser.add_argument("--seed", type=int, default=10_000)
    return parser.parse_args()


def main():
    args = parse_args()
    env = PaperTripleBalanceEnv(
        max_steps=5000,
        reset_range=PaperTripleBalanceEnv.TEST_RESET_RANGE,
    )
    model_class = A2C if args.algorithm == "a2c" else PPO
    model = model_class.load(str(args.model), env=env, device="cpu")
    observation, _ = env.reset(seed=args.seed)

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                started = time.perf_counter()
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(action)
                viewer.sync()

                if terminated or truncated:
                    time.sleep(0.4)
                    observation, _ = env.reset()

                remaining = env.CONTROL_DT - (time.perf_counter() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        env.close()


if __name__ == "__main__":
    main()
