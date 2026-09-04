import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

from cart_triple_pendulum_env import CartTriplePendulumEnv


def validate_model_configuration(env):
    actuator_joint_ids = env.model.actuator_trnid[:, 0]
    hinge_joint_ids = [env._joint_id(f"joint{index}") for index in range(1, 4)]

    assert (env.model.nq, env.model.nv, env.model.nu) == (4, 4, 1)
    assert env.action_space.shape == (1,)
    assert actuator_joint_ids.tolist() == [env._cart_joint_id]
    assert not any(joint_id in actuator_joint_ids for joint_id in hinge_joint_ids)
    assert np.allclose(env.action_space.low, [-10.0])
    assert np.allclose(env.action_space.high, [10.0])
    assert np.allclose(env._cart_range, [-2.5, 2.5])

    down_heights = env._endpoint_heights_at(np.zeros(3))
    up_heights = env._endpoint_heights_at(env.UPRIGHT_HINGE_QPOS)
    assert np.allclose(down_heights, [1.9, 1.1, 0.3], atol=1e-6)
    assert np.allclose(up_heights, [3.5, 4.3, 5.1], atol=1e-6)

    env._set_configuration(0.0, env.UPRIGHT_HINGE_QPOS)
    assert env._state_metrics()["is_balanced"]

    print("Cart model configuration confirmed:")
    print("  one actuator drives cart_slide with force range -10 to 10 N")
    print("  joint1, joint2, and joint3 have no actuators")
    print("  cart travel range is -2.5 to 2.5 m")
    print(f"  hanging endpoint z: {np.round(down_heights, 3)}")
    print(f"  upright endpoint z: {np.round(up_heights, 3)}")


def validate_cart_motion(env):
    env._set_configuration(0.0, np.zeros(3))
    initial_position = float(env.data.qpos[env._cart_qpos_index])
    env.data.ctrl[:] = env.action_space.high
    for _ in range(20):
        mujoco.mj_step(env.model, env.data)
    moved_position = float(env.data.qpos[env._cart_qpos_index])
    assert moved_position > initial_position + 0.01
    print(f"Cart actuation confirmed: x moved from {initial_position:.3f} "
          f"to {moved_position:.3f} m")


def validate_reward_gating(env):
    poses = {
        "down": np.array([0.0, 0.0, 0.0]),
        "only_link1_up": np.array([np.pi, np.pi, 0.0]),
        "only_links1_and2_up": np.array([np.pi, 0.0, np.pi]),
        "all_links_up": env.UPRIGHT_HINGE_QPOS,
    }
    rewards = {}
    for name, hinge_qpos in poses.items():
        env._set_configuration(0.0, hinge_qpos)
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
    maximum_cart_travel = 0.0

    for _ in range(steps):
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        maximum_cart_travel = max(maximum_cart_travel, abs(info["cart_position"]))
        if terminated or truncated:
            observation, _ = env.reset()

    print(f"Random-action smoke test passed: {steps} steps")
    print(f"  maximum absolute cart position: {maximum_cart_travel:.3f} m")


def main():
    env = CartTriplePendulumEnv()
    try:
        validate_model_configuration(env)
        validate_cart_motion(env)
        validate_reward_gating(env)
        check_env(env, warn=True)
        print("Stable-Baselines3 check_env passed")
        run_random_action_smoke_test(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
