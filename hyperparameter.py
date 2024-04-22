import gymnasium as gym
import optuna
import os
import sys
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
import pickle
import pyperbot_v2
import optuna.visualization as vis
import joblib
import json
import logging
import configparser

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
    vf_coef = trial.suggest_uniform('vf_coef', 0.5, 1.0)

    # Create the environment
    env = gym.make(str(args.environment))
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.timesteps)
    trial_monitor_dir = os.path.join('pyperbot_v2/logs/', f'trial_{trial.number + 1}')
    os.makedirs(trial_monitor_dir, exist_ok=True)
    env = Monitor(env, trial_monitor_dir, allow_early_resets=True)
    # Create model
    model = PPO('MlpPolicy', env, learning_rate=learning_rate, batch_size=batch_size,
                gamma=gamma, clip_range=clip_range, ent_coef = ent_coef, vf_coef = vf_coef, tensorboard_log=f'pyperbot_v2/logs/trial_{trial.number+1}')
    model.learn(total_timesteps=int(2e5))

    # Model evaluation
    reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic = False)
    return reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimisation for snakebot")
    parser.add_argument('-e', '--environment', help="Environment to train the model on", default='StandardSnakebotEnv')
    parser.add_argument('-tt', '--timesteps', help="Number of timesteps for the simulation", default=50000, type=int)
    parser.add_argument('-s', '--save_path', help="Path to save the study object", default='study.pkl')
    parser.add_argument('-p', '--plot_results', help="Plot the study results", default = True)
    parser.add_argument('-c', '--clear_logs', help="Clear the logs", default = False)
    parser.add_argument('-cf', '--clear_files', help="Clear the files", default = False)
    args = parser.parse_args()
    

    if args.clear_files:
        # file_clear('pyperbot_v2/logs/actions')
        # file_clear('pyperbot_v2/logs/rewards')
        # os.remove('studyPPO_optimisation12.db')
        os.makedirs('pyperbot_v2/logs/actions', exist_ok=True)
        os.makedirs('pyperbot_v2/logs/rewards', exist_ok=True)

    # add optuna logging
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "PPO_optimisation12"

    #define storage url
    storage_url = f"sqlite:///study{study_name}.db"

    study = optuna.create_study(study_name = study_name, direction="maximize", storage=storage_url, load_if_exists=True)
    with tqdm(total=10, desc="Trials") as pbar:
        for _ in range(10):
            #create trial results folder
            print(f"\n Trial {_+1}/10")
            study.optimize(objective, n_trials=1)
            pbar.update(1)
            #save results of each trial to csv
            df = study.trials_dataframe(attrs = ('number', 'value', 'params'))
            df.to_csv(f'pyperbot_v2/logs/trial_{_+1}/trial{_+1}.csv', index = False)
            #move data from stored config to logs
            # for filename in os.listdir(f'pyperbot_v2/logs/stored_configs'):
            #     os.rename(f'pyperbot_v2/logs/stored_configs/{filename}', f'pyperbot_v2/logs/trial_{_+1}/{filename}')
            #move data from rewards to trials
            for filename in os.listdir(f'pyperbot_v2/logs/rewards'):
                os.rename(f'pyperbot_v2/logs/rewards/{filename}', f'pyperbot_v2/logs/trial_{_+1}/{filename}')

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the study
    with open(args.save_path, 'wb') as f:
        joblib.dump(study, f)
    # Plot the results if requested
    if args.plot_results:
        vis.plot_optimization_history(study)
        vis.plot_param_importances(study)
        vis.plot_parallel_coordinate(study)
        vis.plot_intermediate_values(study)
        vis.plot_slice(study)

        #show plots
        vis.plot_contour(study, params = ['learning_rate', 'batch_size'])
        vis.plot_contour(study, params = ['gamma', 'clip_range'])
        vis.plot_contour(study, params = ['ent_coef', 'vf_coef'])

    # clear the logs at the end if option is enabled
    if args.clear_logs:
        file_clear('pyperbot_v2/logs/')
        print("Logs cleared")

    
