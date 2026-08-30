# Architecture Document

## Digital Twin–Driven RL for Medical Imaging Workflow Optimization

> LJMU Master's Thesis — Developer Reference
>
> ⚠️ **SYNTHETIC / RESEARCH DATA ONLY** — No real patient data.

---

## 1. System Overview

This project implements a **Hierarchical Multi-Agent Reinforcement Learning (MARL)** framework to optimize medical imaging workflows in hospitals. A **Digital Twin** simulates the hospital environment, and RL agents learn to make scheduling decisions that outperform traditional baselines (FIFO, Static Triage).

```mermaid
graph TB
    subgraph "Entry Points"
        CLI["CLI<br/>(medical_imaging_rl/cli.py)"]
        EXP["Experiment Runner<br/>(experiments/runners/full_experiment.py)"]
        DASH["Dashboard<br/>(dashboard/app.py)"]
    end

    subgraph "Core Engine"
        DT["Digital Twin<br/>(simulator/)"]
        AGENTS["RL Agents<br/>(agents/)"]
        RL["RL Core<br/>(rl/)"]
        RW["Reward System<br/>(rewards/)"]
    end

    subgraph "Evaluation & Output"
        EVAL["Evaluation<br/>(evaluation/)"]
        RPT["Reports<br/>(reports/)"]
        BL["Baselines<br/>(baselines/)"]
    end

    CLI --> EXP
    DASH --> DT
    EXP --> DT
    EXP --> BL
    EXP --> RL
    EXP --> EVAL
    EXP --> RPT
    AGENTS --> DT
    RL --> AGENTS
    RL --> RW
    RW --> DT
    EVAL --> RL
    RPT --> EVAL

    style DT fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    style AGENTS fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px
    style RL fill:#FFF3E0,stroke:#E65100,stroke-width:2px
    style RW fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px
```

---

## 2. High-Level Data Flow

The core training loop follows this decision flow at every simulation step:

```mermaid
flowchart TD
    A["🏥 Medical Imaging Event<br/>(Poisson arrival)"] --> B["Digital Twin<br/>(Event-Driven Simulation)"]
    B --> C["Global State S_t<br/>[Q, W, M, G, R, SLA, C]"]
    C --> D["Observation Builders<br/>(Heterogeneous per agent)"]
    D --> E["🧠 Workflow Agent (Level 1)<br/>5 strategies"]
    E -->|"Strategy context<br/>(one-hot)"| F["Level 2 Agents"]

    subgraph F["Level 2 Agents (Parallel Decisions)"]
        F1["📋 Queue Agent<br/>8 actions"]
        F2["🖥️ Resource Agent<br/>60 actions"]
        F3["🤖 AI Model Agent<br/>5 actions"]
    end

    F --> G["Joint Action<br/>(workflow, queue, scanner, radiologist, model)"]
    G --> H["🛡️ Constraint Validator<br/>9 safety constraints"]
    H -->|"Approved / Corrected"| I["Digital Twin Step"]
    I --> J["New State S_t+1"]
    I --> K["Rewards"]

    subgraph K["Reward Calculation"]
        K1["R_global<br/>(7 weighted components)"]
        K2["R_local per agent<br/>(4 agent-specific)"]
        K3["R_i = α·R_local + β·R_global"]
    end

    K --> L["📦 Rollout Buffer<br/>(per agent)"]
    L --> M["HAPPO Sequential Update<br/>(Workflow → Queue → Resource → Model)"]
    M --> E

    style E fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px
    style H fill:#FFCDD2,stroke:#C62828,stroke-width:2px
    style K1 fill:#E1BEE7,stroke:#6A1B9A,stroke-width:2px
    style M fill:#FFE0B2,stroke:#E65100,stroke-width:2px
```

---

## 3. Module Architecture

### 3.1 Project Directory Structure

```
LJMU_MI_WF/
├── config.yaml                              # Master experiment configuration
├── pyproject.toml                           # Python project metadata
│
├── simulator/                               # 🏥 DIGITAL TWIN (Environment)
│   ├── digital_twin/environment.py          #   MedicalImagingDigitalTwin class
│   ├── events/
│   │   ├── event.py                         #   Event & EventType definitions
│   │   ├── event_queue.py                   #   Heap-based priority event queue
│   │   └── event_sources.py                 #   Poisson arrival generator
│   ├── state/
│   │   ├── study.py                         #   Study, Urgency, Modality, StudyStage
│   │   └── global_state.py                  #   7-component GlobalState
│   ├── resources/
│   │   └── resources.py                     #   Scanner, Radiologist, AIModel, GPUCluster
│   ├── scenarios/
│   │   └── scenario.py                      #   6 scenario configurations (A–F)
│   └── transitions/                         #   State transition logic
│
├── agents/                                  # 🧠 RL AGENTS
│   ├── observations.py                      #   4 heterogeneous observation builders
│   ├── workflow/agent.py                    #   Level 1: Workflow Agent (5 actions)
│   ├── queue/agent.py                       #   Level 2: Queue Agent (8 actions)
│   ├── resource/agent.py                    #   Level 2: Resource Agent (60 actions)
│   ├── model_selection/agent.py             #   Level 2: AI Model Agent (5 actions)
│   └── constraint_validator.py              #   9 deterministic safety constraints
│
├── rl/                                      # 📊 REINFORCEMENT LEARNING CORE
│   ├── ppo/
│   │   ├── actor_critic.py                  #   ActorCritic neural network
│   │   ├── buffer.py                        #   RolloutBuffer, MultiAgentRolloutBuffer
│   │   └── gae.py                           #   Generalized Advantage Estimation
│   ├── happo/
│   │   └── happo_trainer.py                 #   HAPPO sequential update trainer
│   ├── policies/                            #   Policy abstractions
│   ├── buffers/                             #   Buffer variants
│   └── training/
│       └── trainer.py                       #   ExperimentTrainer (main training loop)
│
├── rewards/                                 # 🎯 REWARD SYSTEM
│   ├── global_reward.py                     #   7-component weighted global reward
│   ├── workflow_reward.py                   #   Local reward for Workflow Agent
│   ├── queue_reward.py                      #   Local reward for Queue Agent
│   ├── resource_reward.py                   #   Local reward for Resource Agent
│   └── model_reward.py                      #   Local reward for AI Model Agent
│
├── baselines/                               # 📏 NON-LEARNING BASELINES
│   ├── fifo.py                              #   FIFO (first-come-first-served)
│   └── triage.py                            #   Static Triage (Critical > Urgent > Routine)
│
├── evaluation/                              # 📈 EVALUATION FRAMEWORK
│   ├── metrics.py                           #   19 primary/secondary metrics
│   ├── convergence.py                       #   Convergence detection
│   ├── statistics.py                        #   Welch's t-test, Mann-Whitney U, bootstrap CI
│   ├── acceptance.py                        #   Multi-criteria acceptance gate (8 criteria)
│   └── ablation.py                          #   Ablation study runner
│
├── experiments/                             # 🧪 EXPERIMENT PIPELINE
│   ├── runners/full_experiment.py           #   18-step experiment pipeline
│   └── configs/
│       └── quick.yaml                       #   Quick-run configuration
│
├── reports/                                 # 📄 OUTPUT GENERATION
│   ├── chart_generator.py                   #   Plotly charts (HTML + PNG)
│   ├── table_generator.py                   #   Thesis tables (CSV, Excel, Markdown)
│   └── report_generator.py                  #   HTML/Markdown reports
│
├── medical_imaging_rl/                      # 🔌 PACKAGE ENTRY POINT
│   ├── cli.py                               #   CLI with subcommands
│   └── logging.py                           #   Structured logging (structlog)
│
├── dashboard/                               # 🌐 WEB DASHBOARD
│   └── app.py                               #   FastAPI interactive dashboard
│
├── tests/                                   # ✅ TEST SUITE
└── results/                                 # 📂 OUTPUT (generated at runtime)
```

---

## 4. Digital Twin (Simulator)

The Digital Twin is the **environment** in the RL formulation. It is a stateful, discrete-event simulator that models a hospital's medical imaging workflow.

```mermaid
classDiagram
    class MedicalImagingDigitalTwin {
        +ScenarioConfig scenario
        +EventQueue event_queue
        +ResourcePool resource_pool
        +dict all_studies
        +float clock
        +reset() GlobalState
        +step(JointAction) StepResult
        +get_study_candidates() list
        +get_available_scanners() list
        +get_available_radiologists() list
        +get_compatible_models() list
    }

    class GlobalState {
        +QueueState queue
        +WorkflowState workflow
        +ModelState model
        +ComputeState compute
        +HumanResourceState human_resource
        +SLAState sla
        +ClinicalContext clinical
        +to_array() ndarray
    }

    class QueueState {
        +float queue_length
        +float urgent_queue_length
        +float max_wait
        +float mean_wait
        +list priority_distribution
        +float sla_risk
        +to_array() ndarray [8 dims]
    }

    class WorkflowState {
        +float active_studies
        +float completed_studies
        +list processing_stages
        +list modality_distribution
        +to_array() ndarray [10 dims]
    }

    class SLAState {
        +float compliance_rate
        +float studies_at_risk
        +float critical_waiting_time
        +to_array() ndarray [3 dims]
    }

    class ResourcePool {
        +list scanners
        +list radiologists
        +list ai_models
        +GPUCluster gpu_cluster
    }

    class Study {
        +str study_id
        +Urgency urgency
        +Modality modality
        +StudyStage current_stage
        +float arrival_time
        +float turnaround_time
        +bool sla_breached
    }

    MedicalImagingDigitalTwin --> GlobalState : produces
    MedicalImagingDigitalTwin --> ResourcePool : manages
    MedicalImagingDigitalTwin --> Study : tracks
    GlobalState --> QueueState
    GlobalState --> WorkflowState
    GlobalState --> SLAState
```

### Study Lifecycle

```mermaid
stateDiagram-v2
    [*] --> QUEUED : Poisson arrival
    QUEUED --> SCANNING : Scanner assigned
    SCANNING --> AI_ANALYSIS : Scan complete
    AI_ANALYSIS --> REPORTING : AI inference done
    REPORTING --> COMPLETED : Radiologist report done
    COMPLETED --> [*]

    note right of QUEUED
        SLA timer starts here.
        Urgency: Critical / Urgent / Routine
        Modality: CT / MRI / XR / US
    end note

    note right of COMPLETED
        TAT = completion_time - arrival_time
        SLA breached if TAT > threshold
    end note
```

### Scenarios (A–F)

| ID | Name | Arrival Rate | Key Stress Factor |
|----|------|-------------|-------------------|
| A | Normal | 10/hr | Baseline — full resources |
| B | Peak | 20/hr | 2× arrival rate |
| C | High Urgent | 10/hr | 40% critical, 35% urgent |
| D | Low Compute | 10/hr | 50% GPU capacity |
| E | Low Radiologist | 10/hr | 50% radiologist availability |
| F | High Latency | 10/hr | 3× inference latency |

---

## 5. Agent Hierarchy

The system uses a **two-level hierarchical** multi-agent architecture:

```mermaid
graph TB
    subgraph "Level 1 — Strategic"
        WA["🧠 Workflow Agent<br/>obs: 30 dims | actions: 5"]
    end

    subgraph "Level 2 — Tactical (receive workflow context)"
        QA["📋 Queue Agent<br/>obs: 16 dims | actions: 8"]
        RA["🖥️ Resource Agent<br/>obs: 13 dims | actions: 60<br/>(scanners × radiologists)"]
        MA["🤖 AI Model Agent<br/>obs: 14 dims | actions: 5"]
    end

    WA -->|"Strategy one-hot<br/>(5 dims)"| QA
    WA -->|"Strategy one-hot<br/>(5 dims)"| RA
    WA -->|"Strategy one-hot<br/>(5 dims)"| MA

    QA --> JA["Joint Action"]
    RA --> JA
    MA --> JA
    WA --> JA

    JA --> CV["🛡️ Constraint Validator"]
    CV --> DT["Digital Twin Step"]

    style WA fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px
    style QA fill:#BBDEFB,stroke:#1565C0,stroke-width:2px
    style RA fill:#BBDEFB,stroke:#1565C0,stroke-width:2px
    style MA fill:#BBDEFB,stroke:#1565C0,stroke-width:2px
    style CV fill:#FFCDD2,stroke:#C62828,stroke-width:2px
```

### Agent Observations (Heterogeneous)

Each agent sees a **different slice** of the global state:

| Agent | Dims | Components |
|-------|------|------------|
| **Workflow** | 30 | QueueState(8) + HumanResource(4) + ModelState(5) + SLAState(3) + WorkflowState(10) |
| **Queue** | 16 | QueueState(8) + SLAState(3) + WorkflowContext(5) |
| **Resource** | 13 | HumanResource(4) + ComputeState(4) + WorkflowContext(5) |
| **AI Model** | 14 | ModelState(5) + ComputeState(4) + WorkflowContext(5) |

### Constraint Validator (Safety Gate)

The Constraint Validator sits between agent decisions and the Digital Twin. It enforces **9 deterministic safety constraints**:

| # | Constraint | Action on Violation |
|---|-----------|---------------------|
| 1 | Emergency/critical priority cannot be violated | Reject |
| 2 | Unavailable scanner cannot be selected | Reject |
| 3 | Unavailable radiologist cannot be assigned | Reject |
| 4 | Unavailable AI model cannot be selected | Reject |
| 5 | GPU capacity cannot be exceeded | Reject |
| 6 | Memory capacity cannot be exceeded | Reject |
| 7 | Invalid workflow transition cannot occur | Reject |
| 8 | Completed study cannot be reprocessed | Reject |
| 9 | SLA-critical starvation (>30 min) | Force critical study |

---

## 6. RL Core (PPO / HAPPO)

### Neural Network Architecture

Each agent uses an independent **ActorCritic** network with separate actor and critic heads:

```mermaid
graph LR
    subgraph "ActorCritic Network (per agent)"
        OBS["Observation<br/>(variable dims)"]

        subgraph "Actor Head (Policy)"
            A1["Linear → 64"]
            A2["Tanh"]
            A3["Linear → 64"]
            A4["Tanh"]
            A5["Linear → act_dim"]
            A6["Softmax → π(a|s)"]
        end

        subgraph "Critic Head (Value)"
            C1["Linear → 64"]
            C2["Tanh"]
            C3["Linear → 64"]
            C4["Tanh"]
            C5["Linear → 1"]
            C6["V(s)"]
        end

        OBS --> A1 --> A2 --> A3 --> A4 --> A5 --> A6
        OBS --> C1 --> C2 --> C3 --> C4 --> C5 --> C6
    end

    style A6 fill:#C8E6C9,stroke:#2E7D32
    style C6 fill:#BBDEFB,stroke:#1565C0
```

### HAPPO Sequential Update

**HAPPO (Heterogeneous-Agent PPO)** updates agents sequentially with a compound importance ratio M to guarantee monotonic improvement:

```mermaid
sequenceDiagram
    participant B as Rollout Buffer
    participant W as Workflow Agent
    participant Q as Queue Agent
    participant R as Resource Agent
    participant M as AI Model Agent

    Note over B, M: Initialize M = 1 (compound importance ratio)

    B->>W: Batch data (obs, actions, advantages)
    W->>W: PPO clipped surrogate × M
    W-->>B: Update policy, compute ratio_W
    Note over W: M = M × (π_new_W / π_old_W)

    B->>Q: Batch data (obs, actions, advantages)
    Q->>Q: PPO clipped surrogate × M
    Q-->>B: Update policy, compute ratio_Q
    Note over Q: M = M × (π_new_Q / π_old_Q)

    B->>R: Batch data (obs, actions, advantages)
    R->>R: PPO clipped surrogate × M
    R-->>B: Update policy, compute ratio_R
    Note over R: M = M × (π_new_R / π_old_R)

    B->>M: Batch data (obs, actions, advantages)
    M->>M: PPO clipped surrogate × M
    M-->>B: Update policy
```

### Key PPO Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda for advantage estimation |
| `clip_epsilon` | 0.20 | PPO clip range |
| `learning_rate` | 3e-4 | Adam optimizer LR |
| `batch_size` | 256 | Minibatch size |
| `ppo_epochs` | 10 | PPO update epochs per rollout |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `value_coef` | 0.5 | Value loss coefficient |
| `hidden_dim` | 64 | Hidden layer width |

---

## 7. Reward System

### Global Reward Formula

```
R_G = w1·C_SLA + w2·C_Clinical − w3·T_Queue − w4·T_Inference − w5·T_Report − w6·U_Resource − w7·I_Workload
```

All metrics are **normalized to [0, 1]** before weighting.

```mermaid
graph LR
    subgraph "Positive Components"
        SLA["w1=0.25 · SLA Compliance"]
        CLIN["w2=0.15 · Clinical Suitability"]
    end

    subgraph "Penalty Components"
        QD["w3=0.15 · Queue Delay"]
        IL["w4=0.10 · Inference Latency"]
        RD["w5=0.15 · Report Delay"]
        RU["w6=0.10 · Resource Inefficiency"]
        WI["w7=0.10 · Workload Imbalance"]
    end

    SLA --> RG["R_global"]
    CLIN --> RG
    QD -->|"subtract"| RG
    IL -->|"subtract"| RG
    RD -->|"subtract"| RG
    RU -->|"subtract"| RG
    WI -->|"subtract"| RG

    style RG fill:#E1BEE7,stroke:#6A1B9A,stroke-width:2px
```

### Per-Agent Reward Mixing

Each agent receives a **blended reward**:

```
R_agent = α · R_local + β · R_global     (α = 0.6, β = 0.4)
```

| Agent | Local Reward Focus |
|-------|-------------------|
| **Workflow** | Strategy effectiveness, throughput |
| **Queue** | Queue delay reduction, priority adherence |
| **Resource** | Resource utilization balance, scanner efficiency |
| **AI Model** | Clinical suitability, inference speed |

---

## 8. Evaluation Framework

```mermaid
graph TB
    subgraph "Data Collection"
        PPO_R["PPO Results<br/>(per seed, per scenario)"]
        FIFO_R["FIFO Results"]
        TRIAGE_R["Triage Results"]
    end

    subgraph "Analysis Pipeline"
        METRICS["19 Metrics<br/>(metrics.py)"]
        CONV["Convergence Detection<br/>(convergence.py)"]
        STATS["Statistical Tests<br/>(statistics.py)"]
        ABL["Ablation Studies<br/>(ablation.py)"]
    end

    subgraph "Decision Gate"
        ACC["Acceptance Criteria<br/>(acceptance.py)<br/>8 criteria, ALL must pass"]
    end

    subgraph "Output"
        VERDICT["✅ PPO PREFERRED<br/>or<br/>⚠️ NO SUPPORTED WINNER"]
    end

    PPO_R --> METRICS
    FIFO_R --> METRICS
    TRIAGE_R --> METRICS
    PPO_R --> CONV
    METRICS --> STATS
    METRICS --> ABL
    STATS --> ACC
    CONV --> ACC
    ABL --> ACC
    ACC --> VERDICT

    style ACC fill:#FFF9C4,stroke:#F57F17,stroke-width:2px
    style VERDICT fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px
```

### 19 Metrics Tracked

**Primary (7):**
Mean TAT, Median TAT, P95 TAT, Mean Queue Wait, P95 Queue Wait, SLA Compliance, Critical Case Wait

**Secondary (12):**
Radiologist Workload Variance, Resource Utilization, Scanner Utilization, GPU Utilization, AI Inference Latency, SLA Violations, Critical Starvation Count, Throughput, Completed Studies, Rejected Actions, Constraint Violations, Policy Inference Time

### Acceptance Criteria (8 Gates)

PPO is declared "preferred" **ONLY if ALL 8 criteria pass**:

| Gate | Criterion | Threshold |
|------|-----------|-----------|
| A | Global reward improvement | ≥ 5% over best baseline |
| B | No degradation of critical-case priority | Critical delay ≤ baseline |
| C | SLA compliance | ≥ baseline |
| D | Queue delay & TAT improved | ≥ 5% reduction |
| E | Resource utilization acceptable | Balanced workload |
| F | Statistical significance | Welch's t-test / Mann-Whitney U |
| G | Consistent across seeds | ≥ 3 of 5 seeds agree |
| H | Generalizes to unseen scenarios | ≥ 1 unseen scenario |

> **If ANY criterion fails, PPO is NOT declared preferred.** This is critical for research integrity.

### Statistical Methods

| Method | Purpose |
|--------|---------|
| Welch's t-test | Compare means across policies |
| Mann-Whitney U | Non-parametric alternative |
| Bootstrap CI (n=10,000) | 95% confidence intervals |

---

## 9. Experiment Pipeline (18 Steps)

```mermaid
graph TB
    subgraph "Phase 1: Validation [Steps 1-3]"
        S1["Step 1: Validate Config"]
        S2["Step 2: Validate Digital Twin"]
        S3["Step 3: Validate Baselines"]
    end

    subgraph "Phase 2: Baselines [Steps 4-5]"
        S4["Step 4: Run FIFO<br/>(all seeds × scenarios)"]
        S5["Step 5: Run Triage<br/>(all seeds × scenarios)"]
    end

    subgraph "Phase 3: Training [Steps 6-8]"
        S6["Step 6: Train PPO/HAPPO<br/>(all seeds × scenarios)"]
        S7["Step 7: Detect Convergence"]
        S8["Step 8: Save Checkpoints"]
    end

    subgraph "Phase 4: Evaluation [Steps 9-11]"
        S9["Step 9: Evaluate PPO"]
        S10["Step 10: Test Unseen Scenarios"]
        S11["Step 11: Multi-Seed Aggregation"]
    end

    subgraph "Phase 5: Analysis [Steps 12-15]"
        S12["Step 12: Ablation Studies"]
        S13["Step 13: Confidence Intervals"]
        S14["Step 14: Statistical Tests"]
        S15["Step 15: Acceptance Criteria"]
    end

    subgraph "Phase 6: Reporting [Steps 16-18]"
        S16["Step 16: Generate Charts"]
        S17["Step 17: Generate Tables"]
        S18["Step 18: Generate Report"]
    end

    S1 --> S2 --> S3
    S3 --> S4 --> S5
    S5 --> S6 --> S7 --> S8
    S8 --> S9 --> S10 --> S11
    S11 --> S12 --> S13 --> S14 --> S15
    S15 --> S16 --> S17 --> S18

    style S15 fill:#FFF9C4,stroke:#F57F17,stroke-width:2px
    style S18 fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px
```

---

## 10. Web Dashboard

The dashboard is a **FastAPI** application that runs a live experiment on each request and renders interactive Plotly charts.

```mermaid
graph LR
    BROWSER["🌐 Browser"] -->|"GET /"| FASTAPI["FastAPI<br/>(dashboard/app.py)"]
    FASTAPI -->|"Runs live"| LIVE["run_live_experiment()"]
    LIVE --> FIFO["FIFO Baseline"]
    LIVE --> TRIAGE["Triage Baseline"]
    LIVE --> PPO_T["PPO Training<br/>(30 episodes)"]
    LIVE --> SCENARIO["Scenario Comparison"]
    FIFO --> HTML["build_dashboard_html()"]
    TRIAGE --> HTML
    PPO_T --> HTML
    SCENARIO --> HTML
    HTML -->|"Plotly charts +<br/>comparison tables"| BROWSER

    style FASTAPI fill:#BBDEFB,stroke:#1565C0,stroke-width:2px
```

| Endpoint | Description |
|----------|-------------|
| `GET /` | Full interactive dashboard (runs live experiment) |
| `GET /api/health` | Health check (`{"status": "ok"}`) |

---

## 11. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Hierarchical agents** (L1 + L2) | Workflow strategy at L1 provides context to tactical L2 agents, reducing joint action space from multiplicative to additive |
| **Heterogeneous observations** | Each agent only sees relevant state components, improving learning efficiency |
| **HAPPO over independent PPO** | Sequential update with compound importance ratio M provides monotonic joint-policy improvement guarantees |
| **Deterministic constraint validator** | Safety constraints are NOT learned — they are hard-coded gates ensuring clinical safety regardless of RL policy |
| **Separate actor/critic networks** | Prevents gradient interference between policy and value objectives |
| **Blended rewards (α=0.6, β=0.4)** | Local rewards encourage specialization; global component ensures agents cooperate toward system-wide objectives |
| **Acceptance criteria gate** | PPO is NOT hard-coded to win — it must pass 8 rigorous criteria to be declared preferred |
| **Event-driven simulation** | More realistic than fixed-timestep simulation for hospital workflows with stochastic arrivals |

---

## 12. Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python ≥ 3.10 |
| Deep Learning | PyTorch ≥ 2.0 |
| Web Framework | FastAPI + Uvicorn |
| Visualization | Plotly |
| Statistics | SciPy, NumPy |
| Data Handling | Pandas, OpenPyXL |
| Configuration | PyYAML |
| Logging | structlog |
| Serialization | Pydantic |

---

## 13. Dependency Map

```mermaid
graph TD
    CLI["medical_imaging_rl/cli.py"] --> TRAINER["rl/training/trainer.py"]
    EXP["experiments/runners/full_experiment.py"] --> TRAINER
    EXP --> BL_FIFO["baselines/fifo.py"]
    EXP --> BL_TRIAGE["baselines/triage.py"]
    EXP --> EVAL_M["evaluation/metrics.py"]
    EXP --> EVAL_S["evaluation/statistics.py"]
    EXP --> EVAL_A["evaluation/acceptance.py"]
    EXP --> RPT_C["reports/chart_generator.py"]
    EXP --> RPT_T["reports/table_generator.py"]
    EXP --> RPT_R["reports/report_generator.py"]

    TRAINER --> DT["simulator/digital_twin/environment.py"]
    TRAINER --> HAPPO["rl/happo/happo_trainer.py"]
    TRAINER --> AGENTS_W["agents/workflow/agent.py"]
    TRAINER --> AGENTS_Q["agents/queue/agent.py"]
    TRAINER --> AGENTS_R["agents/resource/agent.py"]
    TRAINER --> AGENTS_M["agents/model_selection/agent.py"]
    TRAINER --> CV["agents/constraint_validator.py"]
    TRAINER --> OBS["agents/observations.py"]
    TRAINER --> RW_G["rewards/global_reward.py"]
    TRAINER --> RW_W["rewards/workflow_reward.py"]
    TRAINER --> RW_Q["rewards/queue_reward.py"]
    TRAINER --> RW_R["rewards/resource_reward.py"]
    TRAINER --> RW_M["rewards/model_reward.py"]
    TRAINER --> CONV["evaluation/convergence.py"]

    HAPPO --> AC["rl/ppo/actor_critic.py"]
    HAPPO --> BUF["rl/ppo/buffer.py"]
    HAPPO --> GAE["rl/ppo/gae.py"]

    DT --> EV["simulator/events/"]
    DT --> ST["simulator/state/"]
    DT --> RS["simulator/resources/"]
    DT --> SC["simulator/scenarios/"]

    BL_FIFO --> DT
    BL_TRIAGE --> DT

    DASH["dashboard/app.py"] --> DT
    DASH --> BL_FIFO
    DASH --> BL_TRIAGE
    DASH --> TRAINER

    style TRAINER fill:#FFE0B2,stroke:#E65100,stroke-width:2px
    style DT fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    style HAPPO fill:#FFF9C4,stroke:#F57F17,stroke-width:2px
```

---

*Document generated for developer reference — LJMU Master's Thesis project.*
