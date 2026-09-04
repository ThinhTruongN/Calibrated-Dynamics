from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from triple_pendulum_env import TriplePendulumEnv


project_root = Path(__file__).resolve().parents[2]
checkpoint_directory = project_root / "checkpoints"
checkpoint_directory.mkdir(exist_ok=True)

env = Monitor(TriplePendulumEnv())

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=str(checkpoint_directory),
    name_prefix="sac_v3",
)

model = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    seed=42,
    device="cpu",
    learning_rate=3e-4,
    learning_starts=5_000,
    buffer_size=200_000,
    batch_size=256,
)

print("Starting version 2 training...")

model.learn(
    total_timesteps=100_000,
    callback=checkpoint_callback,
)

model_path = (
    checkpoint_directory
    / "sac_triple_pendulum_v3_100k"
)
model.save(str(model_path))

mean_reward, reward_std = evaluate_policy(
    model,
    env,
    n_eval_episodes=5,
    deterministic=True,
)

print("Training finished!")
print(f"Mean reward: {mean_reward:.2f}")
print(f"Reward standard deviation: {reward_std:.2f}")
print(f"Model saved to: {model_path}.zip")

env.close()
