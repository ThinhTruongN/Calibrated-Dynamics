"""Conformal calibration of multi-step model predictions (README part 4).

The planner trusts the learned model H steps ahead. Here we ask: how far can
the true state be from the model's k-step prediction, with a guarantee?

For a start state s_t and the actions actually applied, the nominal model
predicts s_hat_{t+k} (k = 1..H). The prediction error on the constraint
coordinates c = [cart_x, phi_1, phi_2, phi_3] gives a nonconformity score

    S_k = max_d |s_{t+k,d} - s_hat_{t+k,d}| / scale_{k,d},

where scale_{k,d} fixes the *shape* of the region (per-dimension error spread
on the calibration data, or the ensemble's own predictive std). Split
conformal prediction turns calibration scores into a radius q_k such that the
box s_hat_{t+k} +/- q_k * scale_{k,d} contains the true state with
probability >= 1 - alpha when calibration and test trajectories are
exchangeable. Those boxes are the planner's safety margins.

Exchangeability fails under distribution shift (a heavier pendulum, a weaker
motor). OnlineConformalMargins then adapts each q_k during control from the
errors it actually observes, with a rolling window of recent scores plus
adaptive conformal inference (Gibbs & Candes, 2021).
"""

from bisect import bisect_left, insort
from math import ceil

import numpy as np
import torch

from common import ANGLE_LIMIT, CART_LIMIT

CONSTRAINT_DIMS = [0, 1, 2, 3]


def quantile_of_sorted(sorted_scores, alpha):
    """Finite-sample conformal quantile; +inf if too few scores, 0 if alpha >= 1."""
    n = len(sorted_scores)
    rank = ceil((n + 1) * (1 - alpha))
    if rank > n:
        return float("inf")
    if rank < 1:
        return 0.0
    return float(sorted_scores[rank - 1])


@torch.no_grad()
def mean_rollout(model, start_states, action_sequences):
    """Open-loop prediction with the ensemble-mean dynamics (the nominal model).

    start_states (batch, 8), action_sequences (batch, H) -> (batch, H, 8)
    Also returns the ensemble std of each predicted state (batch, H, 8), the
    spread of the members' own open-loop trajectories.
    """
    s = torch.as_tensor(np.asarray(start_states, dtype=np.float32))
    acts = torch.as_tensor(np.asarray(action_sequences, dtype=np.float32))
    members = model.members
    member_s = s.expand(members, -1, -1).clone()
    means, spreads = [], []
    for k in range(acts.shape[1]):
        a = acts[:, k].reshape(1, -1, 1).expand(members, -1, -1)
        nxt, _ = model(s.expand(members, -1, -1), a)
        s = nxt.mean(0)
        member_s, _ = model(member_s, a)
        means.append(s)
        spreads.append(member_s.std(0))
    return torch.stack(means, 1).numpy(), torch.stack(spreads, 1).numpy()


def kstep_errors(model, dataset, horizon):
    """k-step prediction errors along recorded trajectories.

    Uses the actions actually applied. Returns
      errors  (n_starts, horizon, 4)  true - predicted, NaN past episode end
      spread  (n_starts, horizon, 4)  ensemble std of the prediction
      episode (n_starts,)             episode id of each start (for splits)
    """
    states, actions, nexts, episodes = (dataset["states"], dataset["actions"][:, 0],
                                        dataset["next_states"], dataset["episode"])
    starts, act_seqs, truths, masks, start_eps = [], [], [], [], []
    for ep in np.unique(episodes):
        idx = np.flatnonzero(episodes == ep)
        traj_states = np.vstack([states[idx], nexts[idx[-1]][None]])  # length T + 1
        traj_actions = actions[idx]
        T = len(idx)
        for t in range(T):
            k_avail = min(horizon, T - t)
            seq = np.zeros(horizon)
            seq[:k_avail] = traj_actions[t:t + k_avail]
            truth = np.full((horizon, 8), np.nan)
            truth[:k_avail] = traj_states[t + 1:t + 1 + k_avail]
            starts.append(traj_states[t])
            act_seqs.append(seq)
            truths.append(truth)
            start_eps.append(ep)
    starts, act_seqs, truths = np.array(starts), np.array(act_seqs), np.array(truths)
    pred, spread = [], []
    for chunk in range(0, len(starts), 4096):
        p, s = mean_rollout(model, starts[chunk:chunk + 4096], act_seqs[chunk:chunk + 4096])
        pred.append(p)
        spread.append(s)
    pred, spread = np.concatenate(pred), np.concatenate(spread)
    errors = (truths - pred)[..., CONSTRAINT_DIMS]
    return errors, spread[..., CONSTRAINT_DIMS], np.array(start_eps)


def scores_from_errors(errors, scale):
    """S = max_d |error_d| / scale_d. errors (n, H, 4), scale (H, 4) or (n, H, 4)."""
    return np.max(np.abs(errors) / scale, axis=-1)


class SplitConformalBoxes:
    """Offline split-conformal k-step prediction boxes.

    shape="fixed"     scale_{k,d} = std of calibration errors (same box shape everywhere)
    shape="ensemble"  scale = ensemble std of each prediction (box follows model doubt)
    """

    def __init__(self, alpha=0.1, shape="fixed", floor=1e-4):
        if shape not in ("fixed", "ensemble"):
            raise ValueError("shape must be 'fixed' or 'ensemble'")
        self.alpha, self.shape, self.floor = alpha, shape, floor

    def _scale(self, spread):
        if self.shape == "fixed":
            return self.fixed_scale
        return np.maximum(spread, self.floor)

    def fit(self, errors, spread):
        self.fixed_scale = np.maximum(np.nanstd(errors, axis=0), self.floor)  # (H, 4)
        scores = scores_from_errors(errors, self._scale(spread))  # (n, H)
        self.calibration_scores = [np.sort(s[np.isfinite(s)]) for s in scores.T]
        self.q = np.array([quantile_of_sorted(s, self.alpha) for s in self.calibration_scores])
        return self

    def half_widths(self, spread=None):
        """Box half-widths (n, H, 4) or (H, 4) for the fixed shape."""
        if self.shape == "fixed":
            return self.q[:, None] * self.fixed_scale
        return self.q[None, :, None] * self._scale(spread)

    def coverage(self, errors, spread):
        """Per-horizon coverage and mean half-width on test errors."""
        hw = self.half_widths(spread)
        if hw.ndim == 2:
            hw = np.broadcast_to(hw, errors.shape)
        inside = np.all(np.abs(errors) <= hw, axis=-1)
        valid = np.all(np.isfinite(errors), axis=-1)
        cov = np.array([inside[valid[:, k], k].mean() for k in range(errors.shape[1])])
        width = np.array([np.nanmean(hw[valid[:, k], k], axis=0) for k in range(errors.shape[1])])
        return cov, width


def gaussian_box_coverage(errors, spread, alpha=0.1):
    """Coverage of the uncalibrated ensemble box s_hat +/- z * ensemble_std.

    z makes a 4-dim box with independent Gaussian coordinates contain 1 - alpha:
    each coordinate gets (1 - alpha)^(1/4).
    """
    from scipy.stats import norm

    z = norm.ppf(0.5 + 0.5 * (1 - alpha) ** (1 / errors.shape[-1]))
    inside = np.all(np.abs(errors) <= z * np.maximum(spread, 1e-6), axis=-1)
    valid = np.all(np.isfinite(errors), axis=-1)
    return np.array([inside[valid[:, k], k].mean() for k in range(errors.shape[1])])


class OnlineConformalMargins:
    """Planner safety margins that adapt online to observed k-step errors.

    Starts from offline split-conformal boxes (fixed shape). After every real
    step it scores the predictions the nominal model *would have made* for the
    current state from each of the last H states (using the actions actually
    applied), then for each horizon k:
      - adaptive=False: keeps the offline radius q_k (static conformal margins)
      - adaptive=True:  slides a window of the most recent scores and updates
                        alpha_k by ACI, so q_k tracks the current error level.
                        The window (50 steps = 1 s) and step size (gamma =
                        0.05) are chosen so margins re-adapt within about a
                        second, since a failing controller has little time.
    margins() returns the (H, 4) half-widths for CEMPlanner.set_margins.
    """

    def __init__(self, model, boxes, horizon, adaptive=True, gamma=0.05, window=50, cap_fraction=None):
        self.model, self.horizon, self.adaptive = model, horizon, adaptive
        # Optional cap: never tighten a bound by more than cap_fraction of the
        # failure limit (diagnostic for over-tightening; None = no cap).
        self.cap_fraction = cap_fraction
        self.alpha, self.gamma = boxes.alpha, gamma
        self.scale = boxes.fixed_scale
        self.q_offline = boxes.q.copy()
        self.window = window
        self.windows = [list(s[-window:]) for s in boxes.calibration_scores]
        self.sorted_windows = [sorted(w) for w in self.windows]
        self.next_evict = [0] * horizon
        self.reset()

    def reset(self):
        """Forget the trajectory history (call at the start of every episode).
        The adapted levels and score windows carry over between episodes."""
        self.states, self.actions = [], []
        if not hasattr(self, "alpha_k"):
            self.alpha_k = np.full(self.horizon, self.alpha)
        self.miss_history = []

    def q(self):
        if not self.adaptive:
            return self.q_offline
        return np.array([quantile_of_sorted(self.sorted_windows[k], self.alpha_k[k]) for k in range(self.horizon)])

    def margins(self):
        q = self.q()
        q = np.where(np.isfinite(q), q, 1e6)
        margins = q[:, None] * self.scale
        if self.cap_fraction is not None:
            limits = np.array([CART_LIMIT, ANGLE_LIMIT, ANGLE_LIMIT, ANGLE_LIMIT])
            margins = np.minimum(margins, self.cap_fraction * limits)
        return margins

    def update(self, state, u, next_state):
        self.states.append(state)
        self.actions.append(u)
        self.states = self.states[-self.horizon:]
        self.actions = self.actions[-self.horizon:]
        n = len(self.states)
        starts = np.array(self.states)                    # starts s_{t-n+1} .. s_t
        seqs = np.zeros((n, self.horizon))
        for i in range(n):
            seqs[i, : n - i] = self.actions[i:]
        pred, _ = mean_rollout(self.model, starts, seqs)
        q_now = self.q()
        misses = np.full(self.horizon, np.nan)
        for i in range(n):
            k = n - i                                     # this start is k steps back
            err = next_state[CONSTRAINT_DIMS] - pred[i, k - 1, CONSTRAINT_DIMS]
            score = float(np.max(np.abs(err) / self.scale[k - 1]))
            miss = score > q_now[k - 1]
            misses[k - 1] = miss
            if self.adaptive:
                self.alpha_k[k - 1] += self.gamma * (self.alpha - miss)
                w, sw = self.windows[k - 1], self.sorted_windows[k - 1]
                j = self.next_evict[k - 1]
                del sw[bisect_left(sw, w[j])]
                w[j] = score
                insort(sw, score)
                self.next_evict[k - 1] = (j + 1) % len(w)
        self.miss_history.append(misses)
