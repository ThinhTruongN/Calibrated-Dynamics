"""Shared helpers for the model-based experiments (README parts 2-4).

Everything here builds on the paper-reproduction balance task
(src/paper_reproduction/paper_triple_balance_env.py): a cart-mounted triple
pendulum that starts near upright, is pushed by a bang-bang +/-60 N cart force
at 50 Hz, and fails when the cart leaves +/-5.4 m or any relative angle leaves
+/-27 degrees.

State layout used throughout (the environment's observation):
    [cart_x, phi_1, phi_2, phi_3, cart_v, phi_1_dot, phi_2_dot, phi_3_dot]
Action layout used by the learned model and the planner: u in {-1, +1},
i.e. the cart force divided by 60 N. Environment action index = (u + 1) / 2.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "model_based"
sys.path.insert(0, str(PROJECT_ROOT / "src" / "paper_reproduction"))

from paper_triple_balance_env import PaperTripleBalanceEnv  # noqa: E402

STATE_DIM = 8
ACTION_DIM = 1
LINK_LENGTH = 1.0
ANGLE_LIMIT = PaperTripleBalanceEnv.RELATIVE_ANGLE_THRESHOLD
CART_LIMIT = PaperTripleBalanceEnv.CART_POSITION_THRESHOLD
STATE_NAMES = ["cart_x", "phi_1", "phi_2", "phi_3", "cart_v", "phi_1_dot", "phi_2_dot", "phi_3_dot"]

NOMINAL_PHYSICS = {"link_mass_scale": 1.0, "force_scale": 1.0}


def make_env(link_mass_scale=1.0, force_scale=1.0, max_steps=None, reset_range=None):
    """Paper balance environment with optional physical distribution shift.

    link_mass_scale  multiplies the mass and inertia of all three links
                     (e.g. a heavier pendulum than the one the model was fit on)
    force_scale      multiplies the cart force (e.g. a weaker motor)
    The controller and learned model are never told about these changes.
    """
    kwargs = {}
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    if reset_range is not None:
        kwargs["reset_range"] = reset_range
    env = PaperTripleBalanceEnv(**kwargs)

    if link_mass_scale <= 0 or force_scale <= 0:
        raise ValueError("physics scales must be positive")
    model = env.model
    for index in range(1, 4):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link{index}")
        model.body_mass[body_id] *= link_mass_scale
        model.body_inertia[body_id] *= link_mass_scale
    model.actuator_gear[0, 0] *= force_scale
    mujoco.mj_setConst(model, env.data)
    return env


def action_index(u):
    """Map u in {-1, +1} to the environment's discrete action index."""
    return int(u > 0)


def tip_x(states):
    """Horizontal tip position from states (..., 8).

    Hinge angles are relative, so link i's absolute angle is the cumulative sum
    of phi_1..phi_i, and each 1 m link contributes sin(absolute angle).
    """
    absolute = np.cumsum(states[..., 1:4], axis=-1)
    return states[..., 0] + LINK_LENGTH * np.sin(absolute).sum(axis=-1)


def paper_reward(states):
    """The paper's per-step reward r3 (w_p = 0.7), evaluated on states (..., 8)."""
    return (
        1.0
        - 0.7 * np.abs(tip_x(states)) / CART_LIMIT
        - 0.3 * np.abs(states[..., 3]) / ANGLE_LIMIT
    )


def failed(states):
    """True where a state is outside the paper's termination bounds."""
    return (np.abs(states[..., 0]) > CART_LIMIT) | np.any(np.abs(states[..., 1:4]) > ANGLE_LIMIT, axis=-1)
