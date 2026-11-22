# Calibrated Dynamics

Uncertainty aware learning and control of a simulated triple pendulum using MuJoCo, Gymnasium, PyTorch, and reinforcement learning.

## Project Goal

The long-term objective is to develop and evaluate controllers capable of swinging a three link pendulum upright and balancing it.

The project will compare:

1. Model free reinforcement learning.
2. Learned probabilistic dynamics.
3. Uncertainty aware planning and control.
4. Conformal calibration under distribution shift.

## Current Checkpoint

The following components are operational:

- A single pendulum MuJoCo simulation.
- A three link pendulum with three actuated hinge joints.
- Python control of all three motors.
- A custom Gymnasium environment.
- Nine dimensional observations.
- Three dimensional continuous actions.
- Reward and episode calculations.
- Stable Baselines3 environment validation.
- SAC training and model checkpoint saving.
- Real time playback of a trained policy.
- A MuJoCo reproduction of Manzl et al.'s triple-pendulum balance task, with
  A2C and PPO baselines (see `PAPER_REPRODUCTION.md`).
- A learned probabilistic dynamics model: a bootstrapped ensemble of Gaussian
  neural networks fit to random-action data.
- Model-based controllers built on it: LQR from the learned linearization, and
  model-predictive control with nominal, ensemble, pessimistic, and
  conformal-margin variants.
- Conformal calibration of the model's multi-step predictions, offline and
  online (rolling window + adaptive conformal inference), evaluated under a
  shifted motor.

Parts 2-4 of the project goal are implemented on the balance task. Results are
in `MODEL_BASED_RESULTS.md`.

## System Architecture

```text
SAC neural network
        |
        v
Gymnasium environment
        |
        v
MuJoCo physics simulation
        |
        v
Angles, velocities, and rewards
        |
        +--------> SAC neural network
```

## Source Layout

```text
src/
|-- fixed_pivot/          Original fixed-base experiments
|-- sliding_cart/         Continuous-force cart SAC experiment
|-- paper_reproduction/   Paper-style bang-bang balance experiment
`-- model_based/          Learned dynamics, MPC, and conformal calibration
```

## Running the model-based experiments

Train the baselines first (they are compared against), then run every stage:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
for algo in a2c ppo; do for seed in 1 2 3; do
  .venv/bin/python src/paper_reproduction/train_paper_triple_balance.py --algorithm $algo --seed $seed
done; done
.venv/bin/python src/model_based/run_all.py
```

On Windows, use `.venv\Scripts\python.exe` and backslashes, as in
`PAPER_REPRODUCTION.md`. The full run takes about 2 hours on an 8-core laptop
CPU; closed-loop tests run in parallel and the run resumes where it stopped if
interrupted. Summary tables and figures are written to `results/model_based/`;
large intermediate files go to `outputs/` (not tracked).
