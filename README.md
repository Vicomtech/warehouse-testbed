# Warehouse Testbed for Reinforcement Learning

This repository provides a configurable **warehouse simulation environment** for benchmarking **reinforcement learning (RL)** algorithms.

It models:
- stochastic item arrivals
- dynamic order generation
- constrained agent movement
- action masking (invalid actions filtered)

It also supports:
- reproducible scenario generation
- dataset export (CSV)
- PPO / MaskablePPO training

---

## Installation

```bash
git clone <your-repo-url>
cd warehouse-testbed
pip install -r requirements.txt
```

---

## Quick Start

Run a simple random agent:

```bash
python run_example.py
```

This will:
- initialize the environment
- render the warehouse
- sample valid actions
- print rewards

---

## Environment

Main file:

```
environment/env_storehouse.py
```

Gym-style usage:

```python
state, _ = env.reset()
state, reward, terminated, truncated, info = env.step(action)
```

### Action Masking

```python
mask = env.get_available_actions()
```

---

## Scenario Generation

Enable with:

```python
record_scenario=True
```

This generates:

- `*_config.csv` → layout and parameters  
- `*_steps.csv` → initial state + events  

Useful for:
- reproducibility  
- dataset generation  
- benchmarking  

---

## Benchmark Pipeline

Run:

```bash
python benchmark_pipeline.py
```

This will:
- generate scenarios
- train RL agents
- evaluate performance
- save results in `results_*`

---

## Configuration

Defined in:

```
environment/conf.json
```

You can modify layouts, probabilities and warehouse settings.

---

## Structure

```
warehouse-testbed/
├── README.md
├── requirements.txt
├── benchmark_pipeline.py
├── run_example.py
├── environment/
│   ├── env_storehouse.py
│   └── conf.json
```

---

## Use Cases

- RL benchmarking  
- warehouse optimization  
- action masking research  
- dataset generation  

---

# License

MIT License
