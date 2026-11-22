"""Experiments for README parts 2-4, on the paper's triple-pendulum balance task.

Stages (each caches its outputs under outputs/model_based/ and writes small
CSV summaries to results/model_based/):

  data      collect random-action transitions, fit the main ensemble model
  model     part 2: k-step prediction accuracy and data efficiency
  control   part 3: closed-loop control on nominal physics, compared with the
            paper's PPO/A2C controllers, using the paper's test protocol
  coverage  part 4: coverage of k-step prediction boxes under motor shift
  shift     part 4: closed-loop control under motor shift, including MPC with
            static and online-adaptive conformal safety margins

Closed-loop evaluations run in parallel worker processes, one test per job.
"""

import json
import os
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torch

from collect import controller_episodes, merge, random_episodes
from common import OUTPUT_ROOT, PROJECT_ROOT, action_index, make_env
from conformal_dynamics import (
    OnlineConformalMargins,
    SplitConformalBoxes,
    gaussian_box_coverage,
    kstep_errors,
)
from dynamics import ProbabilisticEnsemble
from mpc import LearnedLQR, PolicyCEMPlanner

RESULTS = PROJECT_ROOT / "results" / "model_based"
MODEL_PATH = OUTPUT_ROOT / "ensemble.pt"
DATA_PATH = OUTPUT_ROOT / "random_data.npz"
BOXES_PATH = OUTPUT_ROOT / "conformal_boxes.npz"
HORIZON = 15
ALPHA = 0.1
TEST_SEED = 10_000          # the paper evaluator's seeds: TEST_SEED + i
FORCE_SCALES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
WORKERS = max(1, (os.cpu_count() or 2) - 1)


def _ensure_dirs():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage: data
# ---------------------------------------------------------------------------

def stage_data(episodes=2000, seed=0):
    """~20k random-action transitions and the main 5-member ensemble."""
    _ensure_dirs()
    data = random_episodes(make_env(), episodes, seed=seed)
    np.savez(DATA_PATH, **data)
    model = ProbabilisticEnsemble()
    t0 = time.time()
    info = model.fit(data["states"], data["actions"], data["next_states"], epochs=60, seed=seed)
    model.save(MODEL_PATH)
    summary = {"transitions": int(len(data["states"])), "episodes": episodes,
               "fit_seconds": time.time() - t0, "best_holdout_mse": info["best_holdout_mse"]}
    (RESULTS / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def load_model():
    return ProbabilisticEnsemble.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Controllers, built inside each worker process
# ---------------------------------------------------------------------------

def build_controller(spec, model):
    """spec: dict(kind=..., ...) -> object with reset() and plan(state) -> u."""
    kind = spec["kind"]
    if kind == "lqr":
        return LearnedLQR(model)
    if kind in ("nominal", "ensemble", "pessimistic"):
        return PolicyCEMPlanner(model, variant=kind, horizon=HORIZON, seed=spec.get("seed", 0))
    if kind in ("conformal_static", "conformal_adaptive", "conformal_adaptive_capped"):
        planner = PolicyCEMPlanner(model, variant="conformal", horizon=HORIZON, seed=spec.get("seed", 0))
        boxes = load_boxes()
        planner.margin_tracker = OnlineConformalMargins(
            model, boxes, HORIZON, adaptive=(kind != "conformal_static"),
            cap_fraction=0.5 if kind == "conformal_adaptive_capped" else None)
        return planner
    if kind in ("ppo", "a2c"):
        from stable_baselines3 import A2C, PPO
        cls = PPO if kind == "ppo" else A2C
        policy = cls.load(spec["path"], device="cpu")

        class SB3Controller:
            def reset(self):
                pass

            def plan(self, state):
                action, _ = policy.predict(state.astype(np.float32), deterministic=True)
                return 1.0 if int(action) == 1 else -1.0
        return SB3Controller()
    raise ValueError(f"unknown controller kind {kind}")


def run_test(args):
    """One closed-loop test. Mirrors the paper evaluator's scoring."""
    spec, force_scale, test_index, steps = args
    torch.set_num_threads(1)
    model = load_model()
    controller = build_controller(spec, model)
    env = make_env(force_scale=force_scale, max_steps=steps, reset_range=0.05)
    obs, _ = env.reset(seed=TEST_SEED + test_index)
    controller.reset()
    tracker = getattr(controller, "margin_tracker", None)
    if tracker is not None:
        tracker.reset()
    final_quarter = int(0.75 * steps)
    error, survived, t0 = 0.0, True, time.time()
    margins_log = []
    for step in range(steps):
        if tracker is not None:
            controller.set_margins(tracker.margins())
        u = controller.plan(obs.astype(np.float64))
        next_obs, _, terminated, _, _ = env.step(action_index(u))
        if tracker is not None:
            tracker.update(obs.astype(np.float64), u, next_obs.astype(np.float64))
            if step % 50 == 0:
                margins_log.append(float(np.max(tracker.margins()[:, 1:4])))
        obs = next_obs
        if step >= final_quarter:
            error = max(error, float(np.max(np.abs(obs[:4]))))
        if terminated:
            survived, error = False, float("inf")
            break
    return {
        "controller": spec["name"], "force_scale": force_scale, "test": test_index, "test_steps": steps,
        "survived": survived, "steps": step + 1, "final_quarter_error": error,
        "success": bool(survived and error <= 0.75),
        "ms_per_step": 1000 * (time.time() - t0) / (step + 1),
        "max_angle_margin": max(margins_log) if margins_log else 0.0,
    }


def run_many(jobs, label):
    """Run closed-loop tests in parallel, resumably.

    Each finished test is appended to outputs/model_based/<label>_tests.jsonl
    immediately, and tests already recorded there are skipped, so an
    interrupted stage resumes where it stopped. Expensive (planning) tests are
    started first so they do not straggle at the end.
    """
    log_path = OUTPUT_ROOT / f"{label}_tests.jsonl"
    key = lambda spec, fs, test, steps: (spec["name"], float(fs), int(test), int(steps))
    done = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            done[(row["controller"], float(row["force_scale"]), int(row["test"]), int(row["test_steps"]))] = row
    todo = [job for job in jobs if key(*job) not in done]
    todo.sort(key=lambda job: job[0]["kind"] in ("lqr", "ppo", "a2c"))
    print(f"[{label}] {len(jobs)} tests ({len(jobs) - len(todo)} already done) on {WORKERS} workers", flush=True)
    t0 = time.time()
    if todo:
        with get_context("spawn").Pool(WORKERS) as pool, log_path.open("a") as log:
            for i, row in enumerate(pool.imap_unordered(run_test, todo)):
                log.write(json.dumps(row) + "\n")
                log.flush()
                done[(row["controller"], float(row["force_scale"]), int(row["test"]), int(row["test_steps"]))] = row
                if (i + 1) % max(1, len(todo) // 20) == 0:
                    print(f"  {i + 1}/{len(todo)} done ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame([done[key(*job)] for job in jobs])


def baseline_specs():
    """PPO and A2C checkpoints from src/paper_reproduction (500k steps, seeds 1-3)."""
    specs = []
    for run_dir in sorted((PROJECT_ROOT / "checkpoints").glob("paper_triple_balance_*")):
        config = json.loads((run_dir / "configuration.json").read_text())
        algo, seed = config["algorithm"], config["seed"]
        final = run_dir / f"{algo}_paper_triple_balance_{config['timesteps']}_steps.zip"
        if final.exists():
            specs.append({"kind": algo, "name": f"{algo.upper()} seed {seed}", "path": str(final)})
    return specs


def summarize(df):
    return (df.groupby(["controller", "force_scale"], sort=False)
              .agg(tests=("survived", "size"), survival=("survived", "mean"),
                   success=("success", "mean"), mean_steps=("steps", "mean"),
                   ms_per_step=("ms_per_step", "mean"))
              .reset_index())


# ---------------------------------------------------------------------------
# Stage: model (part 2)
# ---------------------------------------------------------------------------

def _lqr_success(model, tests=20, steps=2500):
    lqr = LearnedLQR(model)
    ok = 0
    for i in range(tests):
        env = make_env(max_steps=steps, reset_range=0.05)
        obs, _ = env.reset(seed=TEST_SEED + i)
        error, alive = 0.0, True
        for step in range(steps):
            obs, _, term, _, _ = env.step(action_index(lqr.plan(obs.astype(np.float64))))
            if step >= 0.75 * steps:
                error = max(error, float(np.max(np.abs(obs[:4]))))
            if term:
                alive = False
                break
        ok += alive and error <= 0.75
    return ok / tests


def stage_model(sizes=(250, 500, 1000, 2000, 5000, 10000, 20000), model_seeds=(0, 1, 2)):
    """Part 2: how much random data does the learned model need, and how
    accurate are its multi-step predictions?"""
    _ensure_dirs()
    torch.set_num_threads(4)
    data = dict(np.load(DATA_PATH))
    episodes = data["episode"]
    rows = []
    for size in sizes:
        cut = np.searchsorted(np.cumsum(np.bincount(episodes)), size) + 1
        mask = episodes < cut
        for ms in model_seeds:
            model = ProbabilisticEnsemble()
            model.fit(data["states"][mask], data["actions"][mask], data["next_states"][mask],
                      epochs=60 if size >= 2000 else 150, seed=ms, verbose=False)
            rate = _lqr_success(model)
            rows.append({"transitions": int(mask.sum()), "model_seed": ms, "lqr_success": rate})
            print(f"  {int(mask.sum()):6d} transitions, seed {ms}: learned-LQR success {rate:.0%}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "data_efficiency.csv", index=False)

    # k-step accuracy on held-out closed-loop trajectories (learned LQR with 10% random actions).
    model = load_model()
    traj, _ = controller_episodes(make_env(max_steps=1000), LearnedLQR(model), 10, seed=7,
                                  max_steps=1000, epsilon=0.1)
    errors, spread, _ = kstep_errors(model, traj, HORIZON)
    acc = []
    for k in range(HORIZON):
        e, s = errors[:, k], spread[:, k]
        ok = np.all(np.isfinite(e), axis=1)
        for d, name in enumerate(["cart_x", "phi_1", "phi_2", "phi_3"]):
            acc.append({"k": k + 1, "coordinate": name,
                        "rmse": float(np.sqrt(np.mean(e[ok, d] ** 2))),
                        "mean_ensemble_std": float(np.mean(s[ok, d]))})
    pd.DataFrame(acc).to_csv(RESULTS / "kstep_accuracy.csv", index=False)


# ---------------------------------------------------------------------------
# Stage: control (part 3)
# ---------------------------------------------------------------------------

MPC_KINDS = ["nominal", "ensemble", "pessimistic"]


def stage_control(tests_cheap=100, tests_mpc=20, steps=5000):
    """Paper protocol (5000-step tests, final-quarter error <= 0.75).
    Cheap controllers get all 100 paper tests; MPC variants the first 20."""
    _ensure_dirs()
    jobs = []
    for spec in baseline_specs() + [{"kind": "lqr", "name": "Learned LQR"}]:
        jobs += [(spec, 1.0, i, steps) for i in range(tests_cheap)]
    for kind in MPC_KINDS:
        spec = {"kind": kind, "name": f"MPC ({kind})"}
        jobs += [(spec, 1.0, i, steps) for i in range(tests_mpc)]
    df = run_many(jobs, "control")
    df.to_csv(OUTPUT_ROOT / "control_tests.csv", index=False)
    summarize(df).to_csv(RESULTS / "control_nominal.csv", index=False)
    print(summarize(df).to_string(index=False))


# ---------------------------------------------------------------------------
# Stage: coverage (part 4)
# ---------------------------------------------------------------------------

def _closed_loop_data(force_scale, episodes, seed, steps=1000):
    torch.set_num_threads(1)
    model = load_model()
    planner = PolicyCEMPlanner(model, variant="nominal", horizon=HORIZON, seed=seed)
    data, _ = controller_episodes(make_env(force_scale=force_scale, max_steps=steps), planner,
                                  episodes, seed=seed, max_steps=steps, epsilon=0.0,
                                  episode_offset=1000 * seed)
    return force_scale, data


def stage_coverage(calib_episodes=12, test_episodes=6, steps=1000):
    """Calibrate k-step boxes on nominal closed-loop MPC data, then test them
    on held-out nominal data and on data from shifted motors."""
    _ensure_dirs()
    jobs = [(1.0, calib_episodes // 4, 100 + j, steps) for j in range(4)]           # calibration
    jobs += [(fs, test_episodes // 2, 200 + 10 * i + j, steps)
             for i, fs in enumerate(FORCE_SCALES) for j in range(2)]                   # test
    print(f"[coverage] collecting closed-loop data in {len(jobs)} jobs", flush=True)
    with get_context("spawn").Pool(WORKERS) as pool:
        results = pool.starmap(_closed_loop_data, jobs)
    calib = merge(*[d for (fs, d), job in zip(results, jobs) if job[2] < 200])
    tests = {}
    for (fs, d), job in zip(results, jobs):
        if job[2] >= 200:
            tests.setdefault(fs, []).append(d)
    tests = {fs: merge(*ds) for fs, ds in tests.items()}

    model = load_model()
    cal_err, cal_spread, _ = kstep_errors(model, calib, HORIZON)
    fixed = SplitConformalBoxes(ALPHA, "fixed").fit(cal_err, cal_spread)
    ens = SplitConformalBoxes(ALPHA, "ensemble").fit(cal_err, cal_spread)
    np.savez(BOXES_PATH, fixed_scale=fixed.fixed_scale, q=fixed.q,
             **{f"scores_{k}": s for k, s in enumerate(fixed.calibration_scores)})

    rows = []
    for fs, data in sorted(tests.items()):
        err, spread, _ = kstep_errors(model, data, HORIZON)
        cov_fixed, w_fixed = fixed.coverage(err, spread)
        cov_ens, w_ens = ens.coverage(err, spread)
        cov_gauss = gaussian_box_coverage(err, spread, ALPHA)
        online = _online_coverage(model, fixed, data)
        for k in range(HORIZON):
            rows.append({"force_scale": fs, "k": k + 1, "n_starts": int(np.isfinite(err[:, k, 0]).sum()),
                         "ensemble_gaussian": cov_gauss[k], "conformal_fixed": cov_fixed[k],
                         "conformal_ensemble": cov_ens[k], "conformal_online": online[k],
                         "halfwidth_phi_fixed": float(np.mean(w_fixed[k][1:4])),
                         "halfwidth_phi_ensemble": float(np.mean(w_ens[k][1:4]))})
    pd.DataFrame(rows).to_csv(RESULTS / "coverage.csv", index=False)
    print(pd.DataFrame(rows).query("k in [1, 5, 10, 15]").round(3).to_string(index=False))


def _online_coverage(model, boxes, data):
    """Coverage of online-adaptive boxes when the test trajectories are replayed
    in order, each episode starting from the offline calibration."""
    misses = []
    for ep in np.unique(data["episode"]):
        idx = np.flatnonzero(data["episode"] == ep)
        tracker = OnlineConformalMargins(model, boxes, HORIZON, adaptive=True)
        for i in idx:
            tracker.update(data["states"][i], float(data["actions"][i, 0]), data["next_states"][i])
        misses.append(np.array(tracker.miss_history))
    m = np.vstack(misses)
    return 1.0 - np.nanmean(m, axis=0)


class _Boxes:
    """Lightweight stand-in for SplitConformalBoxes loaded from disk."""


def load_boxes():
    blob = np.load(BOXES_PATH)
    boxes = _Boxes()
    boxes.alpha = ALPHA
    boxes.fixed_scale = blob["fixed_scale"]
    boxes.q = blob["q"]
    boxes.calibration_scores = [blob[f"scores_{k}"] for k in range(len(boxes.q))]
    return boxes


# ---------------------------------------------------------------------------
# Stage: shift (part 4)
# ---------------------------------------------------------------------------

SHIFT_KINDS = ["lqr", "nominal", "ensemble", "pessimistic", "conformal_static", "conformal_adaptive"]


def stage_shift(tests=20, steps=2000, force_scales=FORCE_SCALES):
    _ensure_dirs()
    names = {"lqr": "Learned LQR", "conformal_static": "MPC (conformal, static)",
             "conformal_adaptive": "MPC (conformal, adaptive)"}
    jobs = []
    for fs in force_scales:
        for spec in baseline_specs():
            if spec["kind"] == "ppo":
                jobs += [(spec, fs, i, steps) for i in range(tests)]
        for kind in SHIFT_KINDS:
            spec = {"kind": kind, "name": names.get(kind, f"MPC ({kind})")}
            jobs += [(spec, fs, i, steps) for i in range(tests)]
    df = run_many(jobs, "shift")
    df.to_csv(OUTPUT_ROOT / "shift_tests.csv", index=False)
    summarize(df).to_csv(RESULTS / "control_shift.csv", index=False)
    print(summarize(df).pivot(index="controller", columns="force_scale", values="survival").round(2).to_string())


def stage_diagnostic(tests=12, steps=2000, force_scales=(0.5, 1.25, 1.5)):
    """Post-hoc diagnostic (not a tuned main result): adaptive conformal MPC
    with every margin capped at half the failure limit, to test whether
    over-tightened bounds explain the adaptive variant's failures."""
    _ensure_dirs()
    spec = {"kind": "conformal_adaptive_capped", "name": "MPC (conformal, adaptive, capped)"}
    jobs = [(spec, fs, i, steps) for fs in force_scales for i in range(tests)]
    df = run_many(jobs, "diagnostic")
    summarize(df).to_csv(RESULTS / "diagnostic_capped_margins.csv", index=False)
    print(summarize(df).to_string(index=False))
