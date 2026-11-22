"""Transition collection for learning the dynamics model."""

import time

import numpy as np

from common import action_index


def empty_buffer():
    return {"states": [], "actions": [], "next_states": [], "episode": []}


def stack(buffer):
    return {
        "states": np.asarray(buffer["states"], dtype=np.float64).reshape(-1, 8),
        "actions": np.asarray(buffer["actions"], dtype=np.float64).reshape(-1, 1),
        "next_states": np.asarray(buffer["next_states"], dtype=np.float64).reshape(-1, 8),
        "episode": np.asarray(buffer["episode"], dtype=np.int64),
    }


def merge(*datasets):
    return {key: np.concatenate([d[key] for d in datasets]) for key in datasets[0]}


def random_episodes(env, episodes, seed, episode_offset=0):
    """Uniformly random bang-bang actions until the pendulum falls.

    From the near-upright training reset these episodes last about a dozen
    steps, so they cover the states just around upright and the early part of
    every way of falling over.
    """
    rng = np.random.default_rng(seed)
    buffer = empty_buffer()
    for episode in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        while True:
            u = float(rng.choice([-1.0, 1.0]))
            next_obs, _, terminated, truncated, _ = env.step(action_index(u))
            buffer["states"].append(obs)
            buffer["actions"].append(u)
            buffer["next_states"].append(next_obs)
            buffer["episode"].append(episode_offset + episode)
            obs = next_obs
            if terminated or truncated:
                break
    return stack(buffer)


def controller_episodes(env, planner, episodes, seed, max_steps, epsilon=0.0, episode_offset=0,
                        on_step=None, verbose=False):
    """Run the planner closed-loop and record every transition.

    epsilon: probability of replacing the planned action by a random one,
             which widens the data around the controller's own trajectories.
    on_step: optional callback(state, u, next_state) after every step (used by
             online conformal calibration to update its margins).
    Returns (dataset, per-episode statistics).
    """
    rng = np.random.default_rng(seed)
    buffer = empty_buffer()
    stats = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        planner.reset()
        start = time.time()
        steps, survived, max_cart = 0, True, 0.0
        for step in range(max_steps):
            u = planner.plan(obs.astype(np.float64))
            if epsilon > 0 and rng.random() < epsilon:
                u = float(rng.choice([-1.0, 1.0]))
            next_obs, _, terminated, truncated, _ = env.step(action_index(u))
            buffer["states"].append(obs)
            buffer["actions"].append(u)
            buffer["next_states"].append(next_obs)
            buffer["episode"].append(episode_offset + episode)
            if on_step is not None:
                on_step(obs.astype(np.float64), u, next_obs.astype(np.float64))
            obs = next_obs
            steps += 1
            max_cart = max(max_cart, abs(float(obs[0])))
            if terminated:
                survived = False
                break
            if truncated:
                break
        stats.append({"steps": steps, "survived": survived, "max_abs_cart": max_cart,
                      "seconds_per_step": (time.time() - start) / max(steps, 1)})
        if verbose:
            print(f"    episode {episode}: {steps} steps, survived={survived}, "
                  f"{1000 * stats[-1]['seconds_per_step']:.1f} ms/step")
    return stack(buffer), stats
