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
`-- paper_reproduction/   Paper-style bang-bang balance experiment
```
