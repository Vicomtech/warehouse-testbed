import csv
import os
import subprocess
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from tqdm import tqdm

from environment.env_storehouse import Storehouse
from training.training_seeds import mask_fn


# =========================================================
# GLOBAL CONFIGURATION
# =========================================================
PARALLEL = True
LAYOUTS = ["6x6fast", "9x9_+", "9x9_L", "9x9_T", "12x12", "realistic_warehouse"]

TRAIN_SEEDS = [5, 2774, 1000, 3333, 16, 1553, 191, 663, 832, 1208, 29, 973, 289, 91, 11, 9830]
EVAL_SEED = 5
MAX_STEPS = 100
TOTAL_TIMESTEPS = 2_000_000
RESULTS_DIR = "results_parallel" if PARALLEL else "results_individual"

# PPO hyperparameters
DEF_LR = 0.00007896699414596331
DEF_GAMMA = 0.5552374983089595
DEF_VF_COEF = 0.8977510742895823
DEF_CLIP_RANGE = 0.33308694741267186
N_STEPS = 1536
ENT_COEF = 0.07282704768258133

POLICY = "CnnPolicy"


def mask_fn(env):
    """Return the action mask of the current environment state."""
    return env.get_available_actions()


# =========================================================
# ENVIRONMENT FACTORY
# =========================================================
def make_env(layout, seed, record_scenario=False, scenario_path=None):
    """
    Create a Storehouse environment instance.

    Args:
        layout: Layout name defined in conf.json.
        seed: Random seed.
        record_scenario: Whether to export scenario CSV files.
        scenario_path: Output path used for scenario logging.

    Returns:
        Configured environment instance.
    """
    env = Storehouse(
        logname=f"log_{layout}_s{seed:03d}",
        logging=False,
        save_episodes=False,
        conf_name=layout,
        max_steps=MAX_STEPS,
        random_start=True,
        path_reward_weight=0,
        reward_function=0,
        gamma=DEF_GAMMA,
        augment=True,
        seed=seed,
        record_scenario=record_scenario,
        scenario_path=scenario_path,
    )

    if not record_scenario:
        env = ActionMasker(env, mask_fn)

    return env


# =========================================================
# EXPERIMENT UTILITIES
# =========================================================
def generate_experiment_tag():
    """
    Create a unique experiment tag using environment version, timestamp,
    and current Git commit hash.
    """
    try:
        from environment.env_storehouse import ENV_VERSION
    except ImportError:
        ENV_VERSION = "v0"

    date_str = datetime.now().strftime("%Y%m%d_%H%M")

    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode().strip()
    except Exception:
        commit_hash = "nogit"

    tag = f"{ENV_VERSION}_{date_str}_{commit_hash}"
    print(f"Experiment tag: {tag}")
    return tag


# =========================================================
# SCENARIO GENERATION
# =========================================================
def generate_scenario(layout, seed, base_dir):
    """
    Generate one reproducible scenario for a given layout and seed.

    This function creates:
    - one shared configuration CSV per layout
    - one step-level event CSV per seed

    Args:
        layout: Layout name.
        seed: Random seed.
        base_dir: Base output directory for the layout.
    """
    scenario_id = f"{layout}_s{seed:03d}"
    seed_dir = os.path.join(base_dir, f"s{seed:03d}")
    os.makedirs(seed_dir, exist_ok=True)

    config_file_global = os.path.join(base_dir, f"sc_{layout}_multi_config.csv")

    if not os.path.exists(config_file_global):
        print(f"Creating shared configuration file for layout={layout}")

        env_config = make_env(layout, seed, record_scenario=True, scenario_path=base_dir)
        env_config._init_scenario_csv(only_steps=False)
        env_config._log_initial_state()
        env_config.close_scenario_logs()
        env_config.close()

        old_name = os.path.join(base_dir, f"{env_config.idScenario}_config.csv")
        if os.path.exists(old_name):
            os.rename(old_name, config_file_global)
            print(f"Saved shared config file: {config_file_global}")

        temp_steps = os.path.join(base_dir, f"{env_config.idScenario}_steps.csv")
        if os.path.exists(temp_steps):
            os.remove(temp_steps)

    print(f"Generating step-level scenario log for {scenario_id}")
    env = make_env(layout, seed, record_scenario=True, scenario_path=seed_dir)

    env._init_scenario_csv(only_steps=True)
    env._log_initial_state()

    for _ in range(MAX_STEPS):
        action = env.action_space.sample()
        _, _, done, truncated, _ = env.step(action)
        env._log_step_events()

        if done or truncated:
            break

    env.close_scenario_logs()
    env.close()

    local_config = os.path.join(seed_dir, f"sc_{layout}_s{seed:03d}_config.csv")
    if os.path.exists(local_config):
        os.remove(local_config)

    print(f"Scenario generated in {seed_dir}\n")


# =========================================================
# PARALLEL TRAINING
# =========================================================
def train_parallel(layout):
    """
    Train a MaskablePPO model in parallel across multiple seeds.

    Args:
        layout: Layout name.

    Returns:
        Mean evaluation reward.
    """
    print(f"\nTraining layout={layout} in parallel with seeds {TRAIN_SEEDS}")

    scenario_id = f"sc_{layout}_multi"
    base_dir = os.path.join(RESULTS_DIR, scenario_id)
    os.makedirs(base_dir, exist_ok=True)

    for seed in TRAIN_SEEDS:
        generate_scenario(layout, seed, base_dir)

    print("Testing action mask locally before spawning subprocesses...")
    env_test = make_env(layout, TRAIN_SEEDS[0], record_scenario=False)
    mask = env_test.env.get_available_actions()
    print("Mask size:", len(mask))
    print("Valid actions:", mask.sum())
    env_test.close()
    print("Mask test completed.\n")

    def make_env_for_seed(s):
        scenario_path = os.path.join(base_dir, f"s{s:03d}", f"{layout}_s{s:03d}")
        return make_env(layout, s, record_scenario=False, scenario_path=scenario_path)

    vec_env = SubprocVecEnv([lambda s=seed: make_env_for_seed(s) for seed in TRAIN_SEEDS])
    eval_env = make_env(layout, EVAL_SEED, record_scenario=False)

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=50,
        min_evals=2,
        verbose=1,
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(base_dir, "best_model"),
        log_path=os.path.join(base_dir, "eval_logs"),
        eval_freq=10000,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback,
    )

    model = MaskablePPO(
        POLICY,
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join(base_dir, "tensorboard_logs"),
        device="cpu",
        learning_rate=DEF_LR,
        gamma=DEF_GAMMA,
        n_steps=N_STEPS,
        ent_coef=ENT_COEF,
        vf_coef=DEF_VF_COEF,
        clip_range=DEF_CLIP_RANGE,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(os.path.join(base_dir, f"ppo_{scenario_id}.zip"))

    eval_file = os.path.join(base_dir, "eval_logs", "evaluations.npz")
    eval_data = np.load(eval_file)
    mean_reward = float(eval_data["results"].mean())

    print(f"Training completed: {scenario_id} (mean reward: {mean_reward:.2f})\n")
    return mean_reward


# =========================================================
# INDIVIDUAL TRAINING
# =========================================================
def train_individual(layout, seed):
    """
    Train one PPO model for a single layout and seed.

    Args:
        layout: Layout name.
        seed: Random seed.

    Returns:
        Mean evaluation reward.
    """
    scenario_id = f"sc_{layout}_s{seed:03d}"
    base_dir = os.path.join(RESULTS_DIR, scenario_id)
    os.makedirs(base_dir, exist_ok=True)

    generate_scenario(layout, seed, base_dir)

    print(f"\nTraining {scenario_id}")
    env = make_env(layout, seed)
    eval_env = make_env(layout, EVAL_SEED)

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=20,
        min_evals=2,
        verbose=0,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(base_dir, "best_model"),
        log_path=os.path.join(base_dir, "eval_logs"),
        eval_freq=10000,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback,
    )

    model = PPO(
        POLICY,
        env,
        verbose=0,
        tensorboard_log=os.path.join(base_dir, "tensorboard_logs"),
        device="cpu",
        learning_rate=DEF_LR,
        gamma=DEF_GAMMA,
        n_steps=N_STEPS,
        ent_coef=ENT_COEF,
        vf_coef=DEF_VF_COEF,
        clip_range=DEF_CLIP_RANGE,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(os.path.join(base_dir, f"ppo_{scenario_id}.zip"))

    eval_file = os.path.join(base_dir, "eval_logs", "evaluations.npz")
    eval_data = np.load(eval_file)
    mean_reward = float(eval_data["results"].mean())

    print(f"Training completed: {scenario_id} (mean reward: {mean_reward:.2f})")
    return mean_reward


# =========================================================
# RESULTS SUMMARY
# =========================================================
def append_summary(csv_path, layout, seed, mean_reward):
    """
    Append one experiment result to the summary CSV.
    """
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp", "layout", "seed", "mode", "mean_reward"])

        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                layout,
                seed,
                "parallel" if PARALLEL else "individual",
                round(mean_reward, 3),
            ]
        )


# =========================================================
# MAIN PIPELINE
# =========================================================
if __name__ == "__main__":
    experiment_tag = generate_experiment_tag()

    RESULTS_DIR = f"results_{experiment_tag}"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_path = os.path.join(RESULTS_DIR, "summary.csv")

    print(f"Pipeline started ({'PARALLEL' if PARALLEL else 'INDIVIDUAL'})")
    print(f"Results directory: {RESULTS_DIR}\n")

    for layout in tqdm(LAYOUTS, desc="Layouts", colour="cyan"):
        if PARALLEL:
            mean_reward = train_parallel(layout)
            append_summary(summary_path, layout, "multi", mean_reward)
        else:
            for seed in TRAIN_SEEDS:
                mean_reward = train_individual(layout, seed)
                append_summary(summary_path, layout, seed, mean_reward)

    print(f"\nPipeline completed ({'PARALLEL' if PARALLEL else 'INDIVIDUAL'})")
    print(f"Global summary saved to {summary_path}")