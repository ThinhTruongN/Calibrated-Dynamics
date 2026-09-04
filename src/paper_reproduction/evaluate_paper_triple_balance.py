import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, PPO

from paper_triple_balance_env import PaperTripleBalanceEnv


def load_model(model_path, algorithm, env):
    model_class = A2C if algorithm == "a2c" else PPO
    return model_class.load(str(model_path), env=env, device="cpu")


def evaluate_model(model, tests=100, steps=5000, seed=10_000):
    if tests <= 0 or steps <= 0:
        raise ValueError("tests and steps must be positive")

    env = PaperTripleBalanceEnv(
        max_steps=steps,
        reset_range=PaperTripleBalanceEnv.TEST_RESET_RANGE,
    )
    results = []
    final_quarter_start = int(0.75 * steps)

    try:
        for test_index in range(tests):
            observation, _ = env.reset(seed=seed + test_index)
            equation_error = 0.0
            released_code_error = 0.0
            survived = True

            for step_index in range(steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(action)

                if step_index >= final_quarter_start:
                    position = env._position_state()
                    equation_error = max(
                        equation_error, float(np.max(np.abs(position)))
                    )
                    released_code_error = max(
                        released_code_error,
                        abs(float(position[0])),
                        abs(float(position[-1])),
                    )

                if terminated:
                    survived = False
                    equation_error = float("inf")
                    released_code_error = float("inf")
                    break
                if truncated:
                    break

            success = bool(
                survived
                and equation_error <= PaperTripleBalanceEnv.TEST_ERROR_THRESHOLD
            )
            results.append(
                {
                    "success": success,
                    "survived": survived,
                    "equation_error": equation_error,
                    "released_code_error": released_code_error,
                }
            )
    finally:
        env.close()

    finite_equation_errors = [
        result["equation_error"]
        for result in results
        if np.isfinite(result["equation_error"])
    ]
    successes = sum(result["success"] for result in results)
    survivals = sum(result["survived"] for result in results)
    worst_error = max(result["equation_error"] for result in results)
    mean_error = (
        float(np.mean(finite_equation_errors))
        if finite_equation_errors
        else float("inf")
    )

    print("Paper-protocol deterministic evaluation:")
    print(f"  tests: {tests}")
    print(f"  rollout length: {steps} steps ({steps * 0.02:.1f} s)")
    print(f"  survival rate: {survivals / tests:.1%} ({survivals}/{tests})")
    print(f"  success rate: {successes / tests:.1%} ({successes}/{tests})")
    print(f"  mean finite final-quarter error: {mean_error:.4f}")
    print(f"  worst final-quarter error: {worst_error:.4f}")
    print(f"  all tests pass: {successes == tests}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the paper-style triple-pendulum balance controller."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("a2c", "ppo"), default="a2c")
    parser.add_argument("--tests", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=10_000)
    return parser.parse_args()


def main():
    args = parse_args()
    env = PaperTripleBalanceEnv(reset_range=PaperTripleBalanceEnv.TEST_RESET_RANGE)
    try:
        model = load_model(args.model, args.algorithm, env)
        evaluate_model(model, tests=args.tests, steps=args.steps, seed=args.seed)
    finally:
        env.close()


if __name__ == "__main__":
    main()
