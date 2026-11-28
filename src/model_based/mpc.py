"""Model-based controllers built on the learned ensemble (README part 3).

Learned LQR ("lqr")
    Linearize the ensemble-mean dynamics at the upright equilibrium,
        s_{t+1} ~ A s_t + B u_t,
    solve the discrete-time Riccati equation for a quadratic cost
        sum_t s_t' Q s_t + R u_t^2,
    and apply the bang-bang action u = sign(-K s). The cart force enters the
    dynamics affinely, so the drift at u = 0 is taken as the average of the
    model's predictions at u = +1 and u = -1 (the only actions it has seen).
    No planning: one matrix-vector product per step.

Model-predictive control ("nominal", "ensemble", "pessimistic", "conformal")
    Every 20 ms, sample candidate *closed-loop* plans
        u_k = sign(-K s_k + n_k),  k = 0..H-1,
    i.e. the learned LQR policy with perturbations n_k, roll each out through
    the learned model, and score it by
        sum_k s_k' Q s_k  +  penalty * [s_k outside the failure bounds]
        + s_H' P s_H        (LQR cost-to-go as terminal cost).
    The perturbation distribution N(mu_k, sigma_k^2) is refined by the
    cross-entropy method (CEM) and warm-started from the previous step. Only
    the first action of the best plan is applied, and the next step replans
    from the newly observed state. Because candidates react to their own
    predicted states, every member of the ensemble can roll the same plan out
    through its own dynamics.

    Variants differ only in how model uncertainty enters the score:
      nominal      one rollout through the ensemble-mean dynamics
      ensemble     every member rolls out the plan; costs averaged (PETS-style)
      pessimistic  ensemble mean cost + kappa * std of cost across members
      conformal    nominal rollout, failure bounds tightened at step k by a
                   calibrated margin m_k (conformal_dynamics.py, part 4)
"""

import numpy as np
import torch
from scipy.linalg import solve_discrete_are

from common import ANGLE_LIMIT, CART_LIMIT

VARIANTS = ("nominal", "ensemble", "pessimistic", "conformal")
DEFAULT_Q = np.diag([1.0, 10.0, 10.0, 10.0, 0.1, 0.1, 0.1, 0.1])
DEFAULT_R = 1.0


def learned_linearization(model, eps=1e-3):
    """(A, B) of the ensemble-mean dynamics at the upright equilibrium."""
    eye = np.eye(8)
    base = np.vstack([eps * eye, -eps * eye, np.zeros((1, 8))])  # (17, 8)
    states = np.vstack([base, base])                             # (34, 8)
    actions = np.concatenate([np.ones(17), -np.ones(17)])        # (34,)
    mu, _ = model.predict_members(states, actions)
    mean = mu.mean(0).astype(np.float64)
    plus, minus = mean[:17], mean[17:]
    drift = 0.5 * (plus[:16] + minus[:16])
    A = ((drift[:8] - drift[8:16]) / (2 * eps)).T
    B = (0.5 * (plus[16] - minus[16]))[:, None]
    return A, B


class LearnedLQR:
    """Bang-bang LQR from the learned model's linearization."""

    def __init__(self, model, Q=DEFAULT_Q, R=DEFAULT_R):
        self.model = model
        self.Q = Q
        self.R = np.array([[R]], dtype=np.float64)
        if not self.update_linearization(model):
            raise RuntimeError("Failed to compute LQR linearization for initial model")

    def update_linearization(self, model=None):
        """Re-linearize the dynamics at upright and re-solve the discrete ARE.

        Returns True if the new solution is valid and applied, False otherwise.
        """
        if model is not None:
            self.model = model
        try:
            A, B = learned_linearization(self.model)
            P = solve_discrete_are(A, B, self.Q, self.R)
            K = np.linalg.solve(self.R + B.T @ P @ B, B.T @ P @ A)
        except Exception:
            return False
        if not (np.all(np.isfinite(P)) and np.all(np.isfinite(K))):
            return False
        self.A, self.B, self.P, self.K = A, B, P, K
        return True

    def reset(self):
        pass

    def plan(self, state):
        return 1.0 if float(-(self.K @ state)[0]) > 0 else -1.0


class PolicyCEMPlanner:
    def __init__(self, model, variant="nominal", horizon=15, population=128, elites=16,
                 iterations=2, noise_std=1.0, failure_penalty=100.0, kappa=1.0, seed=0):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.model, self.variant = model, variant
        self.horizon, self.population, self.elites = horizon, population, elites
        self.iterations, self.noise_std = iterations, noise_std
        self.failure_penalty, self.kappa = failure_penalty, kappa
        self.lqr = LearnedLQR(model)
        self.K = torch.as_tensor(self.lqr.K[0], dtype=torch.float32)
        self.P = torch.as_tensor(self.lqr.P, dtype=torch.float32)
        self.Qdiag = torch.tensor(np.diag(self.lqr.Q).copy(), dtype=torch.float32)
        self.limits = torch.tensor([CART_LIMIT, ANGLE_LIMIT, ANGLE_LIMIT, ANGLE_LIMIT])
        self.margins = torch.zeros(horizon, 4)
        self.generator = torch.Generator().manual_seed(seed)
        self.reset()

    def update_linearization(self, model=None):
        """Point the planner at an updated model and refresh its LQR base policy and terminal cost."""
        if model is not None:
            self.model = model
        if self.lqr.update_linearization(self.model):
            self.K = torch.as_tensor(self.lqr.K[0], dtype=torch.float32)
            self.P = torch.as_tensor(self.lqr.P, dtype=torch.float32)
            return True
        return False

    def reset(self):
        self.mu = torch.zeros(self.horizon)

    def set_margins(self, margins):
        margins = torch.as_tensor(np.asarray(margins, dtype=np.float32))
        if margins.shape != (self.horizon, 4):
            raise ValueError(f"margins must have shape ({self.horizon}, 4)")
        self.margins = margins

    @torch.no_grad()
    def _costs(self, state, noise):
        """Cost of each perturbation sequence noise (population, horizon)."""
        members, n = self.model.members, noise.shape[0]
        single = self.variant in ("nominal", "conformal")
        s = torch.as_tensor(state, dtype=torch.float32).reshape(1, 1, 8).expand(1 if single else members, n, 8)
        total = torch.zeros(s.shape[0], n)
        first_action = None
        for k in range(self.horizon):
            u = torch.where(-(s @ self.K) + noise[:, k] > 0, 1.0, -1.0).unsqueeze(-1)
            if k == 0:
                first_action = u[0, :, 0]
            if single:
                nxt, _ = self.model(s.expand(members, -1, -1), u.expand(members, -1, -1))
                s = nxt.mean(0, keepdim=True)
            else:
                s, _ = self.model(s, u)
            violation = (s[..., :4].abs() > self.limits - self.margins[k]).any(-1)
            total = total + (s**2 * self.Qdiag).sum(-1) + self.failure_penalty * violation.float()
        total = total + torch.einsum("mni,ij,mnj->mn", s, self.P, s)
        if self.variant == "pessimistic":
            return total.mean(0) + self.kappa * total.std(0), first_action
        return total.mean(0), first_action

    @torch.no_grad()
    def plan(self, state):
        """Return the first action u in {-1, +1} of the best plan found."""
        mu, std = self.mu.clone(), torch.full((self.horizon,), self.noise_std)
        best_cost, best_action = np.inf, None
        for _ in range(self.iterations):
            noise = mu + std * torch.randn(self.population, self.horizon, generator=self.generator)
            noise[0] = 0.0  # always include the unperturbed LQR policy
            costs, first = self._costs(state, noise)
            elite_idx = torch.topk(costs, self.elites, largest=False).indices
            elite = noise[elite_idx]
            mu, std = elite.mean(0), elite.std(0).clamp_min(0.1)
            if float(costs[elite_idx[0]]) < best_cost:
                best_cost, best_action = float(costs[elite_idx[0]]), float(first[elite_idx[0]])
        self.mu = torch.cat([mu[1:], torch.zeros(1)])
        self.last_cost = best_cost
        return best_action
