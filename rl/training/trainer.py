"""Experiment Trainer — orchestrates the complete training and evaluation pipeline.

Decision flow per step:
  Medical Imaging Event -> Digital Twin -> Global State -> Workflow Agent ->
  Queue/Resource/AI Model Agents -> Joint Action -> Constraint Validation ->
  Digital Twin Step -> New State -> Global + Local Reward -> PPO Update
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from simulator.digital_twin.environment import MedicalImagingDigitalTwin, JointAction
from simulator.scenarios.scenario import ScenarioConfig
from agents.workflow.agent import WorkflowAgent
from agents.queue.agent import QueueAgent
from agents.resource.agent import ResourceAgent
from agents.model_selection.agent import AIModelAgent
from agents.constraint_validator import ConstraintValidator
from rl.happo.happo_trainer import HAPPOTrainer, HAPPOConfig, AgentConfig
from rl.ppo.buffer import MultiAgentRolloutBuffer
from rewards.global_reward import GlobalRewardCalculator, RewardWeights
from rewards.workflow_reward import WorkflowLocalReward
from rewards.queue_reward import QueueLocalReward
from rewards.resource_reward import ResourceLocalReward
from rewards.model_reward import ModelLocalReward
from evaluation.metrics import MetricsCalculator
from evaluation.convergence import ConvergenceDetector, ConvergenceConfig, ConvergenceResult
from agents.observations import (
    WorkflowObservationBuilder, QueueObservationBuilder,
    ResourceObservationBuilder, AIModelObservationBuilder,
)


@dataclass
class TrainingConfig:
    num_episodes: int = 500
    eval_interval: int = 50
    eval_episodes: int = 10
    checkpoint_interval: int = 100
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    learning_rate: float = 3e-4
    batch_size: int = 256
    ppo_epochs: int = 10
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 64
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    agent_alpha: float = 0.6
    agent_beta: float = 0.4
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    output_dir: str = "results"
    experiment_name: str = "default"


@dataclass
class EpisodeLog:
    episode: int = 0
    global_reward: float = 0.0
    mean_tat: float = 0.0
    sla_compliance: float = 0.0
    completed_studies: int = 0
    training_stats: dict = field(default_factory=dict)
    timestamp: float = 0.0


class ExperimentTrainer:
    """Orchestrates hierarchical PPO/HAPPO training."""

    def __init__(self, config: TrainingConfig, scenario: ScenarioConfig, seed: int = 42):
        self.config = config
        self.scenario = scenario
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.digital_twin = MedicalImagingDigitalTwin(
            config={"max_steps_per_episode": config.num_episodes},
            scenario=scenario, seed=seed,
        )

        # Agents
        self.workflow_agent = WorkflowAgent()
        self.queue_agent = QueueAgent()
        self.resource_agent = ResourceAgent()
        self.model_agent = AIModelAgent()
        self.constraint_validator = ConstraintValidator()

        # HAPPO
        happo_config = HAPPOConfig(
            gamma=config.gamma, gae_lambda=config.gae_lambda,
            clip_epsilon=config.clip_epsilon, ppo_epochs=config.ppo_epochs,
            batch_size=config.batch_size, entropy_coef=config.entropy_coef,
            value_coef=config.value_coef, max_grad_norm=config.max_grad_norm,
            agent_configs={
                "workflow": AgentConfig("workflow", WorkflowObservationBuilder.OBS_DIM,
                                        WorkflowAgent.NUM_ACTIONS, config.hidden_dim, config.learning_rate),
                "queue": AgentConfig("queue", QueueObservationBuilder.OBS_DIM,
                                     QueueAgent.NUM_ACTIONS, config.hidden_dim, config.learning_rate),
                "resource": AgentConfig("resource", ResourceObservationBuilder.OBS_DIM,
                                        ResourceAgent.NUM_ACTIONS, config.hidden_dim, config.learning_rate),
                "model": AgentConfig("model", AIModelObservationBuilder.OBS_DIM,
                                     AIModelAgent.NUM_ACTIONS, config.hidden_dim, config.learning_rate),
            },
        )
        self.happo = HAPPOTrainer(happo_config)
        self.buffers = MultiAgentRolloutBuffer(["workflow", "queue", "resource", "model"])

        # Rewards
        self.global_reward_calc = GlobalRewardCalculator(config.reward_weights)
        self.workflow_reward = WorkflowLocalReward()
        self.queue_reward = QueueLocalReward()
        self.resource_reward = ResourceLocalReward()
        self.model_reward = ModelLocalReward()

        # Evaluation
        self.metrics_calc = MetricsCalculator()
        self.convergence_detector = ConvergenceDetector(config.convergence)

        # Logging
        self.episode_logs: list = []
        self.training_history: list = []
        self.best_eval_score = float("-inf")
        self.best_checkpoint_path = ""

    def train(self) -> dict:
        output_dir = Path(self.config.output_dir) / self.config.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        conv_result = ConvergenceResult()

        for episode in range(self.config.num_episodes):
            ep_result = self._run_training_episode()

            # HAPPO sequential update
            training_stats = self.happo.sequential_update(self.buffers)
            self.buffers.clear_all()

            log_entry = EpisodeLog(
                episode=episode,
                global_reward=ep_result["total_reward"],
                mean_tat=ep_result["mean_tat"],
                sla_compliance=ep_result["sla_compliance"],
                completed_studies=ep_result["completed_studies"],
                training_stats={n: s.__dict__ if hasattr(s, "__dict__") else {} for n, s in training_stats.items()},
                timestamp=time.time(),
            )
            self.episode_logs.append(log_entry)
            self.training_history.append({
                "episode": episode,
                "reward": ep_result["total_reward"],
                "mean_tat": ep_result["mean_tat"],
                "sla_compliance": ep_result["sla_compliance"],
                "completed": ep_result["completed_studies"],
                **{f"{n}_policy_loss": s.policy_loss if hasattr(s, "policy_loss") else 0
                   for n, s in training_stats.items()},
                **{f"{n}_entropy": s.entropy if hasattr(s, "entropy") else 0
                   for n, s in training_stats.items()},
            })

            # Convergence
            mean_kl = float(np.mean([s.approx_kl for s in training_stats.values()
                                     if hasattr(s, "approx_kl")]))
            conv_result = self.convergence_detector.update(ep_result["total_reward"], mean_kl)

            # Eval
            if (episode + 1) % self.config.eval_interval == 0:
                eval_score = self._evaluate(self.config.eval_episodes)
                if eval_score > self.best_eval_score:
                    self.best_eval_score = eval_score
                    self.best_checkpoint_path = str(output_dir / "best_checkpoint.pt")
                    self.happo.save_checkpoint(self.best_checkpoint_path)

            if (episode + 1) % self.config.checkpoint_interval == 0:
                self.happo.save_checkpoint(str(output_dir / f"checkpoint_ep{episode+1}.pt"))

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{self.config.num_episodes} | "
                      f"Reward: {ep_result['total_reward']:.3f} | "
                      f"TAT: {ep_result['mean_tat']:.1f} | "
                      f"SLA: {ep_result['sla_compliance']:.1%} | "
                      f"Conv: {conv_result.message}")

            if conv_result.converged and episode >= self.config.convergence.min_episodes:
                print(f"\n{'='*50}\nCONVERGED at episode {episode+1}\n{'='*50}\n")
                break

        self.happo.save_checkpoint(str(output_dir / "final_checkpoint.pt"))

        return {
            "convergence_result": conv_result,
            "episode_logs": self.episode_logs,
            "training_history": self.training_history,
            "best_checkpoint": self.best_checkpoint_path,
        }

    def _run_training_episode(self) -> dict:
        state = self.digital_twin.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            # 1. Workflow Agent observation + action (Level 1)
            obs_w = self.workflow_agent.get_observation(state)
            w_res = self.happo.get_actions({"workflow": obs_w})
            w_action = w_res["workflow"]["action"]
            self.workflow_agent.set_strategy(w_action)
            wf_ctx = self.workflow_agent.get_strategy_context()

            # 2. Level 2 observations + actions
            obs_q = self.queue_agent.get_observation(state, wf_ctx)
            obs_r = self.resource_agent.get_observation(state, wf_ctx)
            obs_m = self.model_agent.get_observation(state, wf_ctx)
            l2_res = self.happo.get_actions({"queue": obs_q, "resource": obs_r, "model": obs_m})

            # 3. Construct joint action
            sc_idx, rd_idx = self.resource_agent.decode_action(l2_res["resource"]["action"])
            joint_action = JointAction(
                workflow_action=w_action,
                queue_action=l2_res["queue"]["action"],
                resource_scanner=sc_idx,
                resource_radiologist=rd_idx,
                model_action=l2_res["model"]["action"],
            )

            # 4. Constraint validation
            validated = self.constraint_validator.validate(joint_action, self.digital_twin)
            if validated.corrected_action:
                joint_action = validated.corrected_action

            # 5. Step
            result = self.digital_twin.step(joint_action)
            new_state = result.global_state
            done = result.done

            # 6. Calculate rewards
            g_breakdown = self.global_reward_calc.calculate(
                sla_compliance_rate=new_state.sla.compliance_rate,
                clinical_suitability=new_state.model.mean_suitability,
                mean_queue_delay=new_state.queue.mean_wait * 1440.0,
                mean_inference_latency=new_state.model.mean_inference_latency * 30.0,
                mean_report_delay=new_state.queue.max_wait * 1440.0,
                resource_utilization=1.0 - new_state.human_resource.available_radiologists,
                workload_imbalance=new_state.human_resource.workload_variance,
            )
            g_reward = g_breakdown.global_reward

            lw = self.workflow_reward.calculate(
                throughput_rate=new_state.workflow.completed_studies,
                processing_balance=0.5, sla_compliance=new_state.sla.compliance_rate,
                strategy_consistency=0.5,
            )
            lq = self.queue_reward.calculate(
                queue_delay_reduction=max(0, 1 - new_state.queue.mean_wait),
                urgent_prioritized=True,
                sla_risk_reduction=max(0, 1 - new_state.sla.studies_at_risk),
                starvation_count=0,
                max_wait_time=new_state.queue.max_wait * 1440.0,
            )
            lr = self.resource_reward.calculate(
                allocation_feasible=True,
                scanner_utilization=1.0 - new_state.compute.available_compute,
                radiologist_utilization=1.0 - new_state.human_resource.available_radiologists,
                workload_variance=new_state.human_resource.workload_variance,
                idle_resources=0,
            )
            lm = self.model_reward.calculate(
                model_suitability=new_state.model.mean_suitability,
                inference_time=new_state.model.mean_inference_latency * 30.0,
                compute_feasible=True,
                model_cost=new_state.model.mean_gpu_req,
            )

            alpha, beta = self.config.agent_alpha, self.config.agent_beta
            rewards_map = {
                "workflow": alpha * lw + beta * g_reward,
                "queue": alpha * lq + beta * g_reward,
                "resource": alpha * lr + beta * g_reward,
                "model": alpha * lm + beta * g_reward,
            }

            # 7. Store transitions
            for name, obs, res_dict in [
                ("workflow", obs_w, w_res),
                ("queue", obs_q, l2_res),
                ("resource", obs_r, l2_res),
                ("model", obs_m, l2_res),
            ]:
                self.buffers.add(
                    name,
                    obs=torch.FloatTensor(obs),
                    action=res_dict[name]["action"],
                    log_prob=res_dict[name]["log_prob"],
                    reward=rewards_map[name],
                    value=res_dict[name]["value"],
                    done=done,
                )

            total_reward += g_reward
            state = new_state
            step_count += 1

        # Episode metrics
        completed = self.digital_twin.completed_studies
        tats = [s.turnaround_time for s in completed if s.turnaround_time is not None]
        mean_tat = float(np.mean(tats)) if tats else 0.0
        n_total = len(self.digital_twin.all_studies)
        n_breached = sum(1 for s in self.digital_twin.all_studies.values() if s.sla_breached)
        sla_comp = 1.0 - n_breached / max(1, n_total)

        return {
            "total_reward": total_reward,
            "mean_tat": mean_tat,
            "sla_compliance": sla_comp,
            "completed_studies": len(completed),
            "steps": step_count,
        }

    def _evaluate(self, num_episodes: int = 10) -> float:
        total = 0.0
        for _ in range(num_episodes):
            state = self.digital_twin.reset()
            done = False
            ep_reward = 0.0
            while not done:
                obs_w = self.workflow_agent.get_observation(state)
                w_res = self.happo.get_actions({"workflow": obs_w})
                self.workflow_agent.set_strategy(w_res["workflow"]["action"])
                wf_ctx = self.workflow_agent.get_strategy_context()
                obs_q = self.queue_agent.get_observation(state, wf_ctx)
                obs_r = self.resource_agent.get_observation(state, wf_ctx)
                obs_m = self.model_agent.get_observation(state, wf_ctx)
                l2 = self.happo.get_actions({"queue": obs_q, "resource": obs_r, "model": obs_m})
                sc, rd = self.resource_agent.decode_action(l2["resource"]["action"])
                ja = JointAction(w_res["workflow"]["action"],
                                 l2["queue"]["action"], sc, rd, l2["model"]["action"])
                result = self.digital_twin.step(ja)
                state = result.global_state
                done = result.done
                g = self.global_reward_calc.calculate(
                    state.sla.compliance_rate, state.model.mean_suitability,
                    state.queue.mean_wait * 1440, state.model.mean_inference_latency * 30,
                    state.queue.max_wait * 1440,
                    1 - state.human_resource.available_radiologists,
                    state.human_resource.workload_variance,
                )
                ep_reward += g.global_reward
            total += ep_reward
        return total / max(1, num_episodes)
