# Paper Reproduction: Triple-Pendulum Stabilization

This experiment translates the open-access paper by Manzl et al.,
"Reliability evaluation of reinforcement learning methods for mechanical systems
with increasing complexity," into the project's MuJoCo/Gymnasium stack.

- Article: https://doi.org/10.1007/s11044-023-09960-2
- Original simulator: Exudyn
- This translation: MuJoCo

## What problem the paper solves

The paper stabilizes a cart-mounted triple pendulum that **starts near upright**.
It explicitly treats swing-up as a separate control task. This experiment is
therefore a balance-controller reproduction, not a replacement for the existing
downward-starting SAC swing-up experiment.

The paper is useful here because it isolates a simpler question: once the chain
is near upright, can a learned controller reliably keep it there? A later hybrid
or curriculum controller can use swing-up as stage one and this stabilization
task as stage two.

## Paper-to-code mapping

| Paper setting | Implementation |
| --- | --- |
| Cart mass | 1 kg |
| Link mass | 0.1 kg each |
| Link length | 1 m each |
| Link inertia | Uniform thin rod, `m L^2 / 3` about its joint |
| Joint friction | Zero |
| Control interval | 0.02 s (50 Hz) |
| Action 0 | -60 N cart force |
| Action 1 | +60 N cart force |
| Cart threshold | 5.4 m |
| Relative-angle threshold | `3*pi/20` rad (27 degrees) |
| Maximum episode | 2,048 steps (40.96 s) |
| Training reset | Every position and velocity sampled in `[-0.1, 0.1]` |
| Test reset | Every position and velocity sampled in `[-0.05, 0.05]` |
| Policy network | Two hidden layers of 64 units |
| A2C learning rate | `7e-4` |
| PPO learning rate | `5e-4` |
| PPO rollout length | 2,048 steps |
| Device | CPU |

The article reports about 500,000 steps as sufficient for a reasonable
triple-pendulum A2C/PPO controller. Its released experiment driver is configured
for 2,000,000 steps so the authors can observe behavior after initial learning.
This project's training command defaults to 500,000 and never writes into an
existing checkpoint directory.

## Coordinates and observation

The paper uses minimum relative coordinates:

```text
[cart_x, phi_1, phi_2, phi_3,
 cart_v, phi_1_dot, phi_2_dot, phi_3_dot]
```

The new XML is deliberately built so `phi_1 = phi_2 = phi_3 = 0` is physically
upright. MuJoCo hinge coordinates are relative to the parent body. Consequently,
rotating `phi_1` rotates the entire downstream chain, while rotating `phi_2`
leaves link 1 unchanged and rotates links 2 and 3.

Unlike the full swing-up environment, this local balance task does not need
sine/cosine angle encoding because episodes terminate at 27 degrees and never
cross the angle wraparound.

## Reward

The released triple-link experiment uses the paper's reward `r3` with
`w_p = 0.7`:

```text
r = 1
    - 0.7 * abs(tip_x) / 5.4
    - 0.3 * abs(phi_3) / (3*pi/20)
```

`tip_x` is measured from the physical endpoint site on link 3. The maximum
per-step reward is 1 at the centered upright equilibrium. There is no velocity
penalty or force penalty in the paper's reward. The angle and cart termination
limits prevent the policy from collecting reward in arbitrary configurations.

## Success protocol

The exact evaluation command runs 100 deterministic tests. Each test lasts
5,000 steps, or 100 simulated seconds. Only the final quarter is scored:

```text
error = max over final 1,250 steps of
        max(abs(cart_x), abs(phi_1), abs(phi_2), abs(phi_3))
```

A test succeeds when the pendulum never leaves its termination bounds and the
error is at most 0.75. The controller passes the paper protocol only when all
100 tests succeed. The evaluator also computes the slightly narrower metric
used by the authors' released code, which checks cart position and `phi_3`.

## Commands

Validate the MuJoCo translation:

```powershell
.\.venv\Scripts\python.exe src\paper_reproduction\validate_paper_triple_balance.py
```

Train the paper's default A2C configuration for the reported 500,000 steps:

```powershell
.\.venv\Scripts\python.exe src\paper_reproduction\train_paper_triple_balance.py --algorithm a2c --timesteps 500000
```

Evaluate a resulting checkpoint using the full protocol:

```powershell
.\.venv\Scripts\python.exe src\paper_reproduction\evaluate_paper_triple_balance.py --algorithm a2c --model <checkpoint.zip>
```

Watch it in MuJoCo:

```powershell
.\.venv\Scripts\python.exe src\paper_reproduction\watch_paper_triple_balance.py --algorithm a2c --model <checkpoint.zip>
```

## Reproduction boundary

This is a method-level reproduction, not a bit-for-bit numerical reproduction.
The paper uses Exudyn's implicit generalized-alpha integrator and older versions
of Gym and Stable-Baselines3. This project uses MuJoCo `implicitfast`, Gymnasium,
and the installed Stable-Baselines3 release. The physical parameters, state,
action, reward, reset ranges, limits, network size, training rates, and test
protocol are matched; integrator-level trajectories will differ.
