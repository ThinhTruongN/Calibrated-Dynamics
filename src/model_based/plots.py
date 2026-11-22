"""Figures for the model-based experiments, written to results/model_based/.

Palette (validated colorblind-safe for adjacent pairs):
  blue #2a78d6, aqua #1baf7a, yellow #eda100, red #e34948; gray #898781 for
  reference series and target lines. Every series also has its own marker and
  line style, so identity never relies on color alone.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import ALPHA, HORIZON, RESULTS

BLUE, AQUA, YELLOW, RED, GRAY = "#2a78d6", "#1baf7a", "#eda100", "#e34948", "#898781"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _line(ax, x, y, color, marker, style, label, **kw):
    ax.plot(x, y, color=color, marker=marker, linestyle=style, markersize=8, linewidth=2, label=label, **kw)


def _target(ax, y, text):
    ax.axhline(y, color=GRAY, linestyle="--", linewidth=1.5, zorder=1)
    ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-4, 5),
                textcoords="offset points", ha="right", va="bottom", color=GRAY, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))


def plot_control_nominal():
    df = pd.read_csv(RESULTS / "control_nominal.csv")
    df = df.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(df) + 1.4))
    colors = [RED if c.startswith("A2C") else YELLOW if c.startswith("PPO") else BLUE for c in df["controller"]]
    y = np.arange(len(df))
    ax.barh(y, df["success"], color=colors, height=0.6)
    for yi, (s, n) in enumerate(zip(df["success"], df["tests"])):
        ax.text(s + 0.01, yi, f"{s:.0%}  (n={n})", va="center", fontsize=9, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(df["controller"])
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Success rate, paper protocol (100 s tests, final-quarter error ≤ 0.75)")
    ax.set_title("Balance control on nominal physics")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "control_nominal.png")
    plt.close(fig)


def plot_data_efficiency():
    df = pd.read_csv(RESULTS / "data_efficiency.csv")
    g = df.groupby("transitions")["lqr_success"].agg(["mean", "min", "max"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.fill_between(g["transitions"], g["min"], g["max"], color=BLUE, alpha=0.15, linewidth=0)
    _line(ax, g["transitions"], g["mean"], BLUE, "o", "-", None)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Random-action transitions used to fit the model (log scale)")
    ax.set_ylabel("Learned-LQR success rate\n(mean; band = range over 3 model seeds)")
    ax.set_title("Data efficiency of the learned model")
    fig.tight_layout()
    fig.savefig(RESULTS / "data_efficiency.png")
    plt.close(fig)


def plot_kstep_accuracy():
    df = pd.read_csv(RESULTS / "kstep_accuracy.csv")
    styles = {"cart_x": (BLUE, "o", "-"), "phi_1": (AQUA, "s", "--"), "phi_2": (YELLOW, "^", ":"), "phi_3": (RED, "D", "-.")}
    labels = {"cart_x": "cart x", "phi_1": "φ1", "phi_2": "φ2", "phi_3": "φ3"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for coord, g in df.groupby("coordinate", sort=False):
        c, m, s = styles[coord]
        _line(axes[0], g["k"], g["rmse"], c, m, s, labels[coord])
        _line(axes[1], g["k"], g["rmse"] / g["mean_ensemble_std"], c, m, s, labels[coord])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMSE of k-step prediction (m or rad, log scale)")
    axes[0].set_title("Prediction error grows with horizon")
    axes[0].legend(frameon=False, fontsize=9)
    _target(axes[1], 1.0, "error = ensemble std")
    axes[1].set_ylabel("RMSE / mean ensemble std")
    axes[1].set_title("Ensemble spread vs. actual error")
    for ax in axes:
        ax.set_xlabel("Prediction horizon k (steps of 20 ms)")
    fig.suptitle("Learned dynamics: open-loop accuracy on held-out closed-loop trajectories", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS / "kstep_accuracy.png")
    plt.close(fig)


COVERAGE_STYLES = {
    "ensemble_gaussian": (RED, "x", "--", "Ensemble Gaussian box (uncalibrated)"),
    "conformal_fixed": (BLUE, "o", "-", "Split conformal, fixed shape"),
    "conformal_ensemble": (AQUA, "s", "-.", "Split conformal, ensemble shape"),
    "conformal_online": (YELLOW, "^", ":", "Online conformal (rolling window + ACI)"),
}


def plot_coverage():
    df = pd.read_csv(RESULTS / "coverage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    nominal = df[df["force_scale"] == 1.0]
    for col, (c, m, s, label) in COVERAGE_STYLES.items():
        _line(axes[0], nominal["k"], nominal[col], c, m, s, label)
        g = df[df["k"] == HORIZON]
        _line(axes[1], g["force_scale"], g[col], c, m, s, label)
    for ax in axes:
        _target(ax, 1 - ALPHA, f"target ({1 - ALPHA:.0%})")
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("Coverage of the true state")
    axes[0].set_xlabel("Prediction horizon k (steps)")
    axes[0].set_title("Nominal motor: coverage by horizon")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    axes[1].set_xlabel("Motor force scale (1 = nominal)")
    axes[1].set_title(f"Coverage at horizon k = {HORIZON} under motor shift")
    fig.suptitle("Calibrating the learned model's multi-step predictions", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS / "coverage.png")
    plt.close(fig)


def plot_shift():
    df = pd.read_csv(RESULTS / "control_shift.csv")
    ppo = df[df["controller"].str.startswith("PPO")].groupby("force_scale")["success"].mean()
    get = lambda name: df[df["controller"] == name].set_index("force_scale")["success"]
    panels = [
        ("Model-free vs. model-based", [("PPO (mean of 3 seeds)", ppo, YELLOW, "^", ":"),
                                        ("Learned LQR", get("Learned LQR"), AQUA, "s", "-."),
                                        ("MPC (nominal)", get("MPC (nominal)"), BLUE, "o", "-")]),
        ("Ensemble uncertainty in the planner", [("MPC (nominal)", get("MPC (nominal)"), GRAY, "o", "--"),
                                                 ("MPC (ensemble)", get("MPC (ensemble)"), BLUE, "s", "-"),
                                                 ("MPC (pessimistic)", get("MPC (pessimistic)"), RED, "D", "-.")]),
        ("Conformal safety margins", [("MPC (nominal)", get("MPC (nominal)"), GRAY, "o", "--"),
                                      ("MPC (conformal, static)", get("MPC (conformal, static)"), AQUA, "s", "-."),
                                      ("MPC (conformal, adaptive)", get("MPC (conformal, adaptive)"), BLUE, "^", "-")]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (title, series) in zip(axes, panels):
        for label, s, c, m, st in series:
            _line(ax, s.index, s.values, c, m, st, label)
        ax.set_title(title)
        ax.set_xlabel("Motor force scale (1 = nominal)")
        ax.set_ylim(-0.03, 1.05)
        ax.legend(loc="lower left", frameon=False, fontsize=9)
    axes[0].set_ylabel("Success rate (40 s tests)")
    fig.suptitle("Balance control when the motor differs from training", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS / "control_shift.png")
    plt.close(fig)


def make_all():
    for name, fn in [("control_nominal.csv", plot_control_nominal), ("data_efficiency.csv", plot_data_efficiency),
                     ("kstep_accuracy.csv", plot_kstep_accuracy), ("coverage.csv", plot_coverage),
                     ("control_shift.csv", plot_shift)]:
        if (RESULTS / name).exists():
            fn()
            print(f"  wrote {name.replace('.csv', '.png')}")
