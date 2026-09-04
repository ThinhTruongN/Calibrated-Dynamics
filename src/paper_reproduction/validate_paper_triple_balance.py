import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

from paper_triple_balance_env import PaperTripleBalanceEnv


def validate_configuration(env):
    assert (env.model.nq, env.model.nv, env.model.nu) == (4, 4, 1)
    assert env.action_space.n == 2
    assert np.isclose(env.model.opt.timestep, 0.02)
    assert np.allclose(env.model.actuator_ctrlrange[0], [-60.0, 60.0])
    assert env.model.actuator_trnid[:, 0].tolist() == [env._cart_joint_id]
    assert not any(
        joint_id in env.model.actuator_trnid[:, 0]
        for joint_id in env._hinge_joint_ids
    )

    env._set_state(np.zeros(4))
    endpoint_positions = env.data.site_xpos[env._endpoint_site_ids].copy()
    assert np.allclose(endpoint_positions[:, 0], 0.0, atol=1e-8)
    assert np.allclose(endpoint_positions[:, 2], [1.25, 2.25, 3.25], atol=1e-8)
    assert np.isclose(env._calculate_reward(), 1.0)

    # MuJoCo hinge qpos values are relative rotations. With joint1 rotated,
    # all downstream links rotate; with joint2 rotated, link1 stays vertical.
    env._set_state([0.0, np.pi / 2.0, 0.0, 0.0])
    joint1_pose = env.data.site_xpos[env._endpoint_site_ids].copy()
    assert np.all(np.abs(joint1_pose[:, 0]) > 0.9)
    env._set_state([0.0, 0.0, np.pi / 2.0, 0.0])
    joint2_pose = env.data.site_xpos[env._endpoint_site_ids].copy()
    assert np.isclose(joint2_pose[0, 0], 0.0, atol=1e-8)
    assert np.all(np.abs(joint2_pose[1:, 0]) > 0.9)

    print("Paper model configuration confirmed:")
    print("  qpos [cart, 0, 0, 0] is physically upright")
    print("  hinge qpos values are relative joint angles")
    print("  actions 0/1 apply -60/+60 N only to the cart")
    print("  all three hinge joints are passive")


def validate_action_mapping(env):
    velocities = []
    for action in (0, 1):
        env._set_state(np.zeros(4))
        env.step(action)
        velocities.append(float(env.data.qvel[env._cart_qvel_index]))
    assert velocities[0] < 0.0 < velocities[1]
    print(
        "Bang-bang action mapping confirmed: "
        f"vx(action 0)={velocities[0]:.3f}, "
        f"vx(action 1)={velocities[1]:.3f}"
    )


def validate_reset_and_termination(env):
    for seed in range(20):
        observation, _ = env.reset(seed=seed)
        assert np.all(np.abs(observation) <= env.TRAIN_RESET_RANGE + 1e-7)

    env._set_state(
        [0.0, env.RELATIVE_ANGLE_THRESHOLD + 1e-4, 0.0, 0.0]
    )
    assert env._failed()
    env._set_state([env.CART_POSITION_THRESHOLD + 1e-4, 0.0, 0.0, 0.0])
    assert env._failed()
    print("Reset range and failure thresholds confirmed")


def run_random_action_smoke_test(env, steps=500):
    observation, _ = env.reset(seed=42)
    env.action_space.seed(42)
    episodes = 1

    for _ in range(steps):
        observation, reward, terminated, truncated, _ = env.step(
            env.action_space.sample()
        )
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        if terminated or truncated:
            observation, _ = env.reset()
            episodes += 1

    print(f"Random-action smoke test passed: {steps} steps, {episodes} episodes")


def main():
    env = PaperTripleBalanceEnv()
    try:
        validate_configuration(env)
        validate_action_mapping(env)
        validate_reset_and_termination(env)
        check_env(env, warn=True)
        print("Stable-Baselines3 check_env passed")
        run_random_action_smoke_test(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
