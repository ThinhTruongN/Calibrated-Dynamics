from stable_baselines3.common.env_checker import check_env

from triple_pendulum_env import TriplePendulumEnv


env = TriplePendulumEnv()

check_env(env, warn=True)

env.close()

print("Environment is valid!")