"""Learned probabilistic dynamics: a bootstrapped ensemble of Gaussian MLPs (README part 2).

Each member maps (state, u) to a Gaussian over the state *change*
    s_{t+1} - s_t ~ N(mu(s_t, u_t), diag(sigma^2(s_t, u_t)))
and is trained by Gaussian negative log-likelihood on its own bootstrap
resample of the data (Chua et al., 2018, "PETS"). Spread *between* members
measures epistemic uncertainty (what the data do not pin down); each member's
sigma measures aleatoric noise. The simulator is deterministic, so here the
aleatoric part is small and the epistemic part dominates.

All members are evaluated in one batched pass: weights have shape
(members, in, out), so predicting for every member and every candidate plan
is a single batched matrix multiply.
"""

import numpy as np
import torch
from torch import nn

from common import ACTION_DIM, STATE_DIM


class EnsembleLinear(nn.Module):
    def __init__(self, members, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(members, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(members, 1, out_features))
        bound = 1.0 / np.sqrt(in_features)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x):  # x: (members, batch, in)
        return torch.baddbmm(self.bias, x, self.weight)


class ProbabilisticEnsemble(nn.Module):
    def __init__(self, members=5, hidden=200, layers=3):
        super().__init__()
        self.members = members
        in_dim = STATE_DIM + ACTION_DIM
        sizes = [in_dim] + [hidden] * layers
        self.hidden_layers = nn.ModuleList(
            EnsembleLinear(members, a, b) for a, b in zip(sizes[:-1], sizes[1:])
        )
        self.output = EnsembleLinear(members, hidden, 2 * STATE_DIM)
        # Soft bounds on the log-variance, as in PETS, keep NLL training stable.
        self.max_logvar = nn.Parameter(torch.full((1, 1, STATE_DIM), 0.5))
        self.min_logvar = nn.Parameter(torch.full((1, 1, STATE_DIM), -10.0))
        for name in ("in_mean", "in_std", "out_mean", "out_std"):
            size = in_dim if name.startswith("in") else STATE_DIM
            self.register_buffer(name, torch.zeros(size) if "mean" in name else torch.ones(size))

    def set_normalization(self, inputs, targets):
        self.in_mean.copy_(torch.as_tensor(inputs.mean(0), dtype=torch.float32))
        self.in_std.copy_(torch.as_tensor(inputs.std(0) + 1e-6, dtype=torch.float32))
        self.out_mean.copy_(torch.as_tensor(targets.mean(0), dtype=torch.float32))
        self.out_std.copy_(torch.as_tensor(targets.std(0) + 1e-6, dtype=torch.float32))

    def _forward_normalized(self, x):
        """x: (members, batch, in) raw inputs -> normalized-target mean, logvar."""
        h = (x - self.in_mean) / self.in_std
        for layer in self.hidden_layers:
            h = nn.functional.silu(layer(h))
        out = self.output(h)
        mean, logvar = out[..., :STATE_DIM], out[..., STATE_DIM:]
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar

    def forward(self, states, actions):
        """states (members, batch, 8), actions (members, batch, 1) ->
        (mean next state, variance of next state), both (members, batch, 8)."""
        mean, logvar = self._forward_normalized(torch.cat([states, actions], dim=-1))
        delta_mean = mean * self.out_std + self.out_mean
        variance = torch.exp(logvar) * self.out_std**2
        return states + delta_mean, variance

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, states, actions, next_states, epochs=60, batch_size=256, lr=1e-3,
            weight_decay=1e-5, holdout_fraction=0.1, seed=0, verbose=True):
        """Fit every member by Gaussian NLL on its own bootstrap resample.

        Keeps the parameters from the epoch with the lowest held-out MSE.
        Returns a dict of training diagnostics.
        """
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        inputs = np.concatenate([states, actions], axis=1).astype(np.float32)
        targets = (next_states - states).astype(np.float32)
        self.set_normalization(inputs, targets)

        n = len(inputs)
        perm = rng.permutation(n)
        n_hold = max(1, int(holdout_fraction * n))
        hold, train = perm[:n_hold], perm[n_hold:]
        boot = np.stack([rng.choice(train, size=len(train), replace=True) for _ in range(self.members)])

        x_all = torch.as_tensor(inputs)
        y_all = (torch.as_tensor(targets) - self.out_mean) / self.out_std
        x_hold = x_all[hold].expand(self.members, -1, -1)
        y_hold = y_all[hold].expand(self.members, -1, -1)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        best = (np.inf, None)
        history = []
        for epoch in range(epochs):
            self.train()
            order = rng.permuted(boot, axis=1)
            for start in range(0, order.shape[1], batch_size):
                idx = torch.as_tensor(order[:, start:start + batch_size])
                x, y = x_all[idx], y_all[idx]
                mean, logvar = self._forward_normalized(x)
                nll = (((mean - y) ** 2) * torch.exp(-logvar) + logvar).mean()
                loss = nll + 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.eval()
            with torch.no_grad():
                mean, _ = self._forward_normalized(x_hold)
                hold_mse = float(((mean - y_hold) ** 2).mean())
            history.append(hold_mse)
            if hold_mse < best[0]:
                best = (hold_mse, {k: v.clone() for k, v in self.state_dict().items()})
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch:3d}  held-out normalized MSE {hold_mse:.5f}")
        self.load_state_dict(best[1])
        self.eval()
        return {"best_holdout_mse": best[0], "holdout_mse_history": history}

    # ------------------------------------------------------------------
    # Numpy conveniences for evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_members(self, states, actions):
        """One-step prediction from every member.

        states (batch, 8), actions (batch,) or (batch, 1) in {-1, +1}
        -> member means (members, batch, 8), member variances (members, batch, 8)
        """
        s = torch.as_tensor(np.asarray(states, dtype=np.float32)).expand(self.members, -1, -1)
        a = torch.as_tensor(np.asarray(actions, dtype=np.float32).reshape(-1, 1)).expand(self.members, -1, -1)
        mean, var = self(s, a)
        return mean.numpy(), var.numpy()

    @torch.no_grad()
    def rollout(self, start_states, action_sequences):
        """Open-loop k-step predictions, each member following its own trajectory.

        start_states (batch, 8), action_sequences (batch, horizon) in {-1, +1}
        -> member trajectories (members, batch, horizon, 8)
           predictive variance (members, batch, horizon, 8): each member's
           aleatoric variance accumulated along its own trajectory
        """
        s = torch.as_tensor(np.asarray(start_states, dtype=np.float32)).expand(self.members, -1, -1).clone()
        acts = torch.as_tensor(np.asarray(action_sequences, dtype=np.float32))
        aleatoric = torch.zeros_like(s)
        trajectory, variances = [], []
        for k in range(acts.shape[1]):
            a = acts[:, k].reshape(1, -1, 1).expand(self.members, -1, -1)
            s, var = self(s, a)
            aleatoric = aleatoric + var
            trajectory.append(s)
            variances.append(aleatoric)
        return torch.stack(trajectory, dim=2).numpy(), torch.stack(variances, dim=2).numpy()

    def save(self, path):
        torch.save({"state_dict": self.state_dict(), "members": self.members,
                    "hidden": self.output.weight.shape[1], "layers": len(self.hidden_layers)}, path)

    @classmethod
    def load(cls, path):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(members=blob["members"], hidden=blob["hidden"], layers=blob["layers"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model
