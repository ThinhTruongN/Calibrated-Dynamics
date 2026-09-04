from triple_pendulum_env import TriplePendulumEnv


env = TriplePendulumEnv()

observation, info = env.reset(seed=42)
env.action_space.seed(42)

print("Observation shape:", observation.shape)
print("Initial observation:", observation)
print("Action shape:", env.action_space.shape)

for step in range(10):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step {step + 1}: "
        f"reward={reward:.3f}, "
        f"upright_score={info['upright_score']:.3f}"
    )

    if terminated or truncated:
        observation, info = env.reset()

env.close()
