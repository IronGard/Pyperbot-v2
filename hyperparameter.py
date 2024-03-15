import gymnasium as gym
import optuna
import os
import sys
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
import joblib
import pyperbot_v2
import optuna.visualization as vis

# Adding path to pyperbot_v2
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

def objective(trial):
    # Tune four main parameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 256])
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)

    # Create the environment
    env = gym.make(str(args.environment))
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.timesteps)
    env = Monitor(env, 'pyperbot_v2/logs/', allow_early_resets=True)

    # Create model
    model = PPO('MlpPolicy', env, learning_rate=learning_rate, batch_size=batch_size,
                gamma=gamma, clip_range=clip_range, tensorboard_log='pyperbot_v2/logs/')
    model.learn(total_timesteps=int(1e5))

    # Model evaluation
    reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    return reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimisation for snakebot")
    parser.add_argument('-e', '--environment', help="Environment to train the model on", default='LoaderSnakebotEnv')
    parser.add_argument('-tt', '--timesteps', help="Number of timesteps for the simulation", default=2400, type=int)
    parser.add_argument('-s', '--save_path', help="Path to save the study object", default='study.pkl')
    parser.add_argument('-p', '--plot_results', help="Plot the study results", action='store_true')
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    with tqdm(total=50, desc="Trials") as pbar:
        for _ in range(50):
            print(f"Trial {_+1}/50")
            study.optimize(objective, n_trials=1)
            pbar.update(1)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the study
    joblib.dump(study, args.save_path)
    print(f"Study saved to {args.save_path}")

    # Plot the results if requested
    if args.plot_results:
        vis.plot_optimization_history(study)
        vis.plot_param_importances(study)
        vis.plot_parallel_coordinate(study)
        vis.plot_slice(study)