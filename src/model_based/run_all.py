"""Run the model-based experiments (README parts 2-4).

Examples:
    python src/model_based/run_all.py                       # every stage, in order
    python src/model_based/run_all.py --stages control      # one stage
    python src/model_based/run_all.py --stages shift --tests 12

Stages must run in order the first time: data -> model -> control -> coverage
-> shift (coverage writes the conformal boxes that the shift stage uses).
`control` and `shift` need the PPO/A2C checkpoints from
src/paper_reproduction/train_paper_triple_balance.py.
"""

import argparse

import experiments

STAGES = ["data", "model", "control", "coverage", "shift", "plots"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--tests", type=int, default=12, help="tests per controller and motor scale in the shift stage")
    args = parser.parse_args()

    for stage in STAGES:
        if stage not in args.stages:
            continue
        print(f"\n===== stage: {stage} =====", flush=True)
        if stage == "data":
            experiments.stage_data()
        elif stage == "model":
            experiments.stage_model()
        elif stage == "control":
            experiments.stage_control()
        elif stage == "coverage":
            experiments.stage_coverage()
        elif stage == "shift":
            experiments.stage_shift(tests=args.tests, force_scales=[0.5, 1.0, 1.25, 1.5])
        elif stage == "plots":
            import plots
            plots.make_all()


if __name__ == "__main__":
    main()
