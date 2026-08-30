#!/usr/bin/env python3
"""CLI entry point for the medical imaging RL application.

Usage:
    python -m medical_imaging_rl.cli run-experiment --config config.yaml
    python -m medical_imaging_rl.cli run-experiment --config experiments/configs/quick.yaml
    python -m medical_imaging_rl.cli train --seed 42 --scenario normal
    python -m medical_imaging_rl.cli baseline --policy both
"""

import argparse
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Digital Twin-Driven RL for Medical Imaging Workflow Optimization",
    )
    sub = parser.add_subparsers(dest="command")

    # run-experiment
    exp = sub.add_parser("run-experiment", help="Run full experiment pipeline")
    exp.add_argument("--config", default="config.yaml")

    # train
    tr = sub.add_parser("train", help="Train PPO/HAPPO only")
    tr.add_argument("--config", default="config.yaml")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--scenario", default="normal")

    # baseline
    bl = sub.add_parser("baseline", help="Run baseline policies")
    bl.add_argument("--policy", choices=["fifo", "triage", "both"], default="both")
    bl.add_argument("--config", default="config.yaml")
    bl.add_argument("--seed", type=int, default=42)
    bl.add_argument("--scenario", default="normal")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run-experiment":
        from experiments.runners.full_experiment import run_full_experiment
        run_full_experiment(args.config)

    elif args.command == "train":
        _run_train(args)

    elif args.command == "baseline":
        _run_baselines(args)


def _run_train(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    from rl.training.trainer import ExperimentTrainer, TrainingConfig
    from simulator.scenarios.scenario import ScenarioFactory
    from rewards.global_reward import RewardWeights
    from evaluation.convergence import ConvergenceConfig

    ppo_cfg = config.get("ppo", {})
    tr_cfg = config.get("training", {})
    rw_cfg = config.get("rewards", {})
    cv_cfg = config.get("convergence", {})

    tc = TrainingConfig(
        num_episodes=tr_cfg.get("num_episodes", 500),
        eval_interval=tr_cfg.get("eval_interval", 50),
        eval_episodes=tr_cfg.get("eval_episodes", 10),
        checkpoint_interval=tr_cfg.get("checkpoint_interval", 100),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_epsilon=ppo_cfg.get("clip_epsilon", 0.20),
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        batch_size=ppo_cfg.get("batch_size", 256),
        ppo_epochs=ppo_cfg.get("ppo_epochs", 10),
        entropy_coef=ppo_cfg.get("entropy_coef", 0.01),
        value_coef=ppo_cfg.get("value_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        hidden_dim=ppo_cfg.get("hidden_dim", 64),
        reward_weights=RewardWeights(**rw_cfg.get("weights", {})),
        agent_alpha=rw_cfg.get("agent_alpha", 0.6),
        agent_beta=rw_cfg.get("agent_beta", 0.4),
        convergence=ConvergenceConfig(
            window_size=cv_cfg.get("window_size", 100),
            min_improvement=cv_cfg.get("min_improvement", 0.005),
            patience=cv_cfg.get("patience", 5),
            min_episodes=cv_cfg.get("min_episodes", 200),
        ),
        output_dir=config.get("experiment", {}).get("output_dir", "results"),
        experiment_name=f"train_{args.scenario}_seed{args.seed}",
    )
    scenario = ScenarioFactory.create(args.scenario)
    trainer = ExperimentTrainer(tc, scenario, args.seed)
    result = trainer.train()
    print(f"\nTraining complete. Convergence: {result['convergence_result'].message}")
    print(f"Best checkpoint: {result['best_checkpoint']}")


def _run_baselines(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    from simulator.digital_twin.environment import MedicalImagingDigitalTwin
    from simulator.scenarios.scenario import ScenarioFactory
    from baselines.fifo import FIFOPolicy
    from baselines.triage import TriagePriorityPolicy

    scenario = ScenarioFactory.create(args.scenario)

    if args.policy in ("fifo", "both"):
        dt = MedicalImagingDigitalTwin(
            config=config.get("digital_twin", {}), scenario=scenario, seed=args.seed
        )
        result = FIFOPolicy().run_episode(dt)
        m = result["metrics"]
        print(f"FIFO | Completed: {result['completed_studies']} | "
              f"TAT: {m.mean_turnaround_time:.1f} | SLA: {m.sla_compliance:.1%} | "
              f"Breaches: {result['sla_breaches']}")

    if args.policy in ("triage", "both"):
        dt = MedicalImagingDigitalTwin(
            config=config.get("digital_twin", {}), scenario=scenario, seed=args.seed
        )
        result = TriagePriorityPolicy().run_episode(dt)
        m = result["metrics"]
        print(f"Triage | Completed: {result['completed_studies']} | "
              f"TAT: {m.mean_turnaround_time:.1f} | SLA: {m.sla_compliance:.1%} | "
              f"Breaches: {result['sla_breaches']}")


if __name__ == "__main__":
    main()
