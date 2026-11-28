"""Conformal-triggered online model adaptation under distribution shift.

In part 4 (conformal_dynamics.py), offline split-conformal boxes hold ~90%
coverage on nominal physics but collapse immediately to ~0% under a shifted
motor. Using online conformal margins as hard planner constraints made MPC
fail by over-tightening. Here we instead use conformal coverage as a *trigger*
to adapt the dynamics model itself on recent transitions, so the physical shift
is learned rather than merely detected.

Two adaptation mechanisms are compared under the same conformal trigger:
  1. ResidualAdaptedEnsemble ("residual")
     Fits a ridge-regularized linear residual delta_s_res(s, u) = [s, u, 1] W
     on top of the frozen base ensemble (< 1 ms per refit). Stronger ridge on
     state/bias than action avoids collinearity overfitting on short closed-loop
     windows while still allowing state-dependent shifts to be learned as data
     accumulate.
  2. FinetunedEnsembleAdapter ("finetune")
     Fine-tunes the ProbabilisticEnsemble weights on recent transitions using
     normalized MSE loss plus an L2 anchor toward the nominal weights theta_0
     to prevent catastrophic forgetting outside the visited closed-loop states.

After every adaptation, the controller refreshes its upright linearization
(A, B) and discrete Riccati policy (K, P), and the conformal miss window is
cleared so future refits only occur if the adapted model still fails nominal
conformal coverage.
"""

import copy
import time
from collections import deque

import numpy as np
import torch
from torch import nn

from common import STATE_DIM
from conformal_dynamics import CONSTRAINT_DIMS


class ResidualAdaptedEnsemble(nn.Module):
    """Frozen ProbabilisticEnsemble plus an online-fitted linear residual."""

    def __init__(self, base_model, ridge_state=5.0, ridge_action=1e-3, ridge_bias=5.0):
        super().__init__()
        self.base_model = base_model
        self.ridge_state = ridge_state
        self.ridge_action = ridge_action
        self.ridge_bias = ridge_bias
        self.register_buffer("W", torch.zeros(STATE_DIM + 2, STATE_DIM, dtype=torch.float32))

    @property
    def members(self):
        return self.base_model.members

    def reset_weights(self):
        self.W.zero_()

    def forward(self, states, actions):
        base_mean, base_var = self.base_model(states, actions)
        ones = torch.ones_like(actions)
        phi = torch.cat([states, actions, ones], dim=-1)
        return base_mean + phi @ self.W, base_var

    @torch.no_grad()
    def predict_members(self, states, actions):
        s = torch.as_tensor(np.asarray(states, dtype=np.float32)).expand(self.members, -1, -1)
        a = torch.as_tensor(np.asarray(actions, dtype=np.float32).reshape(-1, 1)).expand(self.members, -1, -1)
        mean, var = self(s, a)
        return mean.numpy(), var.numpy()

    def adapt(self, states, actions, next_states):
        """Closed-form ridge regression on 1-step base-model residuals."""
        base_pred, _ = self.base_model.predict_members(states, actions)
        residuals = np.asarray(next_states, dtype=np.float64) - base_pred.mean(0).astype(np.float64)
        phi = np.column_stack([
            np.asarray(states, dtype=np.float64),
            np.asarray(actions, dtype=np.float64).reshape(-1, 1),
            np.ones((len(states), 1), dtype=np.float64),
        ])
        reg = np.diag([self.ridge_state] * STATE_DIM + [self.ridge_action, self.ridge_bias])
        W = np.linalg.solve(phi.T @ phi + reg, phi.T @ residuals)
        self.W.copy_(torch.as_tensor(W, dtype=torch.float32))


class FinetunedEnsembleAdapter:
    """Anchored online fine-tuning of a ProbabilisticEnsemble."""

    def __init__(self, base_model, steps=30, lr=1e-3, anchor_weight=1e-3):
        self.base_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        self.model = copy.deepcopy(base_model)
        self.model.eval()
        self.steps = steps
        self.lr = lr
        self.anchor_weight = anchor_weight
        self._param_names = [
            name for name, _ in self.model.named_parameters()
            if name not in ("max_logvar", "min_logvar")
        ]

    def reset_weights(self):
        self.model.load_state_dict(self.base_state)
        self.model.eval()

    def adapt(self, states, actions, next_states):
        """Fine-tune from the nominal weights on recent transitions with L2 anchoring.

        Resetting to base_state before each refit on the rolling buffer keeps
        updates deterministic and prevents cumulative overfitting across refits.
        """
        self.model.load_state_dict(self.base_state)
        self.model.train()

        inputs = np.concatenate(
            [np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32).reshape(-1, 1)],
            axis=1,
        )
        targets = (np.asarray(next_states, dtype=np.float32) - np.asarray(states, dtype=np.float32))
        x = torch.as_tensor(inputs).expand(self.model.members, -1, -1)
        y = ((torch.as_tensor(targets) - self.model.out_mean) / self.model.out_std).expand(
            self.model.members, -1, -1
        )

        params = [p for name, p in self.model.named_parameters() if name in self._param_names]
        anchors = [self.base_state[name] for name in self._param_names]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        for _ in range(self.steps):
            mean, _ = self.model._forward_normalized(x)
            mse = ((mean - y) ** 2).mean()
            anchor_loss = sum(((p - p0) ** 2).sum() for p, p0 in zip(params, anchors))
            loss = mse + self.anchor_weight * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

        self.model.eval()


class ConformalModelAdapter:
    """Monitor 1-step conformal coverage and adapt the dynamics model on shift.

    Uses the offline split-conformal 1-step box (scale[0], q[0]) calibrated on
    nominal physics as a validity certificate. Whenever empirical 1-step miss
    rate over the rolling window exceeds `miss_threshold`, the dynamics model
    adapts on the recent transition buffer and the controller refreshes its
    linearization (K, P).
    """

    def __init__(
        self,
        base_model,
        boxes,
        controller,
        method="residual",
        window=20,
        min_window=12,
        miss_threshold=0.50,
        buffer_size=200,
        cooldown=15,
        adapter_kwargs=None,
    ):
        if method not in ("residual", "finetune"):
            raise ValueError(f"method must be 'residual' or 'finetune', got {method!r}")
        self.base_model = base_model
        self.method = method
        self.controller = controller
        self.scale_1step = np.asarray(boxes.fixed_scale[0], dtype=np.float64)
        self.q_1step = float(boxes.q[0])
        self.window = int(window)
        self.min_window = int(min_window)
        self.miss_threshold = float(miss_threshold)
        self.buffer_size = int(buffer_size)
        self.cooldown = int(cooldown)

        kwargs = adapter_kwargs or {}
        if method == "residual":
            self.adapter = ResidualAdaptedEnsemble(base_model, **kwargs)
            self.adapted_model = self.adapter
        else:
            self.adapter = FinetunedEnsembleAdapter(base_model, **kwargs)
            self.adapted_model = self.adapter.model

        self.reset()

    def reset(self):
        """Reset model weights, controller linearization, buffers, and logs."""
        self.adapter.reset_weights()
        self.controller.update_linearization(self.adapted_model)
        self._miss_window = deque(maxlen=self.window)
        self._states = deque(maxlen=self.buffer_size)
        self._actions = deque(maxlen=self.buffer_size)
        self._next_states = deque(maxlen=self.buffer_size)
        self._step = 0
        self._steps_since_refit = self.cooldown
        self.refit_steps = []
        self.refit_times_ms = []
        self.miss_history = []
        self.score_history = []
        self.error_1step_history = []

    def update(self, state, u, next_state):
        """Score the 1-step prediction of the current model and adapt if coverage fails."""
        state = np.asarray(state, dtype=np.float64)
        next_state = np.asarray(next_state, dtype=np.float64)
        pred_members, _ = self.adapted_model.predict_members(state[None], np.array([u]))
        pred = pred_members.mean(0)[0]

        err = next_state[CONSTRAINT_DIMS] - pred[CONSTRAINT_DIMS]
        score = float(np.max(np.abs(err) / self.scale_1step))
        miss = bool(score > self.q_1step)

        self._miss_window.append(miss)
        self._states.append(state)
        self._actions.append(float(u))
        self._next_states.append(next_state)

        self.miss_history.append(miss)
        self.score_history.append(score)
        self.error_1step_history.append(float(np.sqrt(np.mean(err ** 2))))
        self._step += 1
        self._steps_since_refit += 1

        triggered = (
            self._steps_since_refit >= self.cooldown
            and len(self._miss_window) >= self.min_window
            and (sum(self._miss_window) / len(self._miss_window)) > self.miss_threshold
        )
        if triggered:
            t0 = time.perf_counter()
            self.adapter.adapt(
                np.asarray(self._states),
                np.asarray(self._actions),
                np.asarray(self._next_states),
            )
            self.controller.update_linearization(self.adapted_model)
            elapsed_ms = 1000.0 * (time.perf_counter() - t0)
            self.refit_steps.append(self._step)
            self.refit_times_ms.append(elapsed_ms)
            self._miss_window.clear()
            self._steps_since_refit = 0
            return True
        return False
