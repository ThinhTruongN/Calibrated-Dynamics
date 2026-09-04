import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

from underactuated_triple_pendulum_env import UnderactuatedTriplePendulumEnv


def validate_model_configuration(env):
    actuator_joint_ids = env.model.actuator_trnid[:, 0]
    joint1_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, "joint1"
    )

    assert env.model.nu == 1
    assert env.action_space.shape == (1,)
    assert actuator_joint_ids.tolist() == [joint1_id]
    assert np.allclose(env.action_space.low, [-2.0])
    assert np.allclose(env.action_space.high, [2.0])

    env.data.qpos[:] = [-np.pi / 2.0, 0.0, 0.0]
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    horizontal_gravity_torque = abs(float(env.data.qfrc_bias[0]))
    assert env.action_space.high[0] < horizontal_gravity_torque

    down_heights = env._endpoint_heights_at(np.zeros(3))
    up_heights = env._endpoint_heights_at(env.UPRIGHT_QPOS)
    assert np.all(up_heights > down_heights)
    assert np.allclose(down_heights, [1.9, 1.1, 0.3], atol=1e-6)
    assert np.allclose(up_heights, [3.5, 4.3, 5.1], atol=1e-6)

    env.data.qpos[:] = env.UPRIGHT_QPOS
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    upright_metrics = env._state_metrics()
    assert upright_metrics["is_balanced"]
    assert np.isclose(upright_metrics["min_individual_link_score"], 1.0)
    assert np.isclose(upright_metrics["endpoint_height_score"], 1.0)

    print("MuJoCo angle convention confirmed:")
    print(f"  qpos=[0, 0, 0] endpoint z: {np.round(down_heights, 3)}")
    print(f"  qpos=[pi, 0, 0] endpoint z: {np.round(up_heights, 3)}")
    print("  joint1 is actuated; joint2 and joint3 are passive")
    print(
        f"  base motor torque range: {env.action_space.low[0]:.1f} to "
        f"{env.action_space.high[0]:.1f} N m"
    )
    print(f"  horizontal-chain gravity load: {horizontal_gravity_torque:.3f} N m")


def validate_reward_gating(env):
    poses = {
        "down": np.array([0.0, 0.0, 0.0]),
        "only_link1_up": np.array([np.pi, np.pi, 0.0]),
        "only_links1_and2_up": np.array([np.pi, 0.0, np.pi]),
        "all_links_up": env.UPRIGHT_QPOS,
    }
    rewards = {}

    for name, qpos in poses.items():
        env.data.qpos[:] = qpos
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        metrics = env._state_metrics()
        rewards[name], _ = env._calculate_reward(np.zeros(1), metrics)

    assert rewards["down"] < 0.01
    assert rewards["only_link1_up"] < 1.0
    assert rewards["only_links1_and2_up"] < 1.0
    assert rewards["all_links_up"] > 10.0

    print("Partial-pose reward gating confirmed:")
    for name, reward in rewards.items():
        print(f"  {name}: {reward:.3f}")


def run_random_action_smoke_test(env, steps=250):
    observation, _ = env.reset(seed=42)
    env.action_space.seed(42)
    peak_min_link_score = 0.0
    peak_endpoint_height_score = 0.0
    episodes = 1

    for _ in range(steps):
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        peak_min_link_score = max(
            peak_min_link_score, info["min_individual_link_score"]
        )
        peak_endpoint_height_score = max(
            peak_endpoint_height_score, info["endpoint_height_score"]
        )

        if terminated or truncated:
            episodes += 1
            observation, _ = env.reset()

    print(f"Random-action smoke test passed: {steps} steps, {episodes} episode(s)")
    print(f"  peak minimum individual-link score: {peak_min_link_score:.3f}")
    print(f"  peak endpoint-height score: {peak_endpoint_height_score:.3f}")


def main():
    env = UnderactuatedTriplePendulumEnv()
    try:
        validate_model_configuration(env)
        validate_reward_gating(env)
        check_env(env, warn=True)
        print("Stable-Baselines3 check_env passed")
        run_random_action_smoke_test(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
