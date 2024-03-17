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


#import clear file from utils
from pyperbot_v2.utils.utils import file_clear

def objective(trial):
    # Tune four main parameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 256])
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-1) #regularisation hyperparameters
    vf_coef = trial.suggest_uniform('vf_coef', 0.1, 1.0)

    # Create the environment
    env = gym.make(str(args.environment))
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.timesteps)
    trial_monitor_dir = os.path.join('pyperbot_v2/logs/', f'trial_{trial.number}')
    os.makedirs(trial_monitor_dir, exist_ok=True)
    env = Monitor(env, trial_monitor_dir, allow_early_resets=True)
    # Create model
    model = PPO('MlpPolicy', env, learning_rate=learning_rate, batch_size=batch_size,
                gamma=gamma, clip_range=clip_range, ent_coef = ent_coef, vf_coef = vf_coef, tensorboard_log='pyperbot_v2/logs/')
    model.learn(total_timesteps=int(1e5))

    # Model evaluation
    reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    return reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimisation for snakebot")
    parser.add_argument('-e', '--environment', help="Environment to train the model on", default='LoaderSnakebotEnv')
    parser.add_argument('-tt', '--timesteps', help="Number of timesteps for the simulation", default=20000, type=int)
    parser.add_argument('-s', '--save_path', help="Path to save the study object", default='study.pkl')
    parser.add_argument('-p', '--plot_results', help="Plot the study results", action='store_true')
    parser.add_argument('-c', '--clear_logs', help="Clear the logs", default = True)
    args = parser.parse_args()
    
    #define storage url
    storage_url = "sqlite:///study.db"

    study = optuna.create_study(direction="maximize", storage=storage_url)
    with tqdm(total=20, desc="Trials") as pbar:
        for _ in range(20):
            print(f"Trial {_+1}/20")
            study.optimize(objective, n_trials=1)
            pbar.update(1)
            #save results of each trial to csv
        df = study.trials_dataframe()
        df.to_csv(f'pyperbot_v2/results/PPO/csv/trial{_+1}.csv', index = False)


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

    # clear the logs at the end if option is enabled
    if args.clear_logs:
        file_clear('pyperbot_v2/logs/')
        print("Logs cleared")