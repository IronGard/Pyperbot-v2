import pybullet as p
import os
import time
import pybullet_data
import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import argparse
import csv
import configparser

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from utils.structure_loader import Loader
from utils.gaits import Gaits
from utils.snakebot_info import Info
from utils.server_code import Server

#Directories
results_dir = "pyperbot_v2/results/"


# Declarables
csv_file_path = 'pyperbot_v2/results/csv/joint_positions.csv'
transmitting_data = []
host = '0.0.0.0'
port = 6969

def main(args):
    pyb_setup = Loader("--mp4=results/videos/training_video.mp4")

    # Setup base plane
    pyb_setup.plane()
    
    # Setup simulation environment
    if args.env == "lab":
        pyb_setup.lab("pyperbot_v2/snakebot_description/meshes/lab_floor_plan.stl")
    elif args.env == "maze":
        pyb_setup.maze("pyperbot_v2/snakebot_description/meshes/maze_10x10.stl")
        goal_positions = pyb_setup.goal(args.num_goals)
    else:
        print('No selected environment.')
        if args.load_config:
            goal_config = configparser.ConfigParser()
            goal_config.read(f'pyperbot_v2/config/seeded_run_configs/goal_config/seed{args.seed}_config.ini')
            sections = goal_config.sections()
            print(sections)
            goal_positions = [[int(x) for x in goal_config['GOALS']['goal1'].split(',')], 
                              [int(x) for x in goal_config['GOALS']['goal2'].split(',')], 
                              [int(x) for x in goal_config['GOALS']['goal3'].split(',')]]
            pyb_setup.manual_goal(len(goal_positions), goal_positions)
    # Setup robot
    if args.load_config:
        snake_config = configparser.ConfigParser()
        snake_config.read(f'pyperbot_v2/config/seeded_run_configs/snake_config/seed{args.seed}_config.ini')
        robot_id = pyb_setup.robot("pyperbot_v2/snakebot_description/urdf/snakebot.urdf.xacro",
                                    basePosition = [float(x) for x in snake_config['SNAKEBOT']['basePosition'].split(',')],
                                    baseOrientation = [float(x) for x in snake_config['SNAKEBOT']['baseOrientation'].split(',')]
                                    )

    # Obtain robot information
    info = Info(robot_id)
    revolute_df, prismatic_df, _, _ = info.joint_info()
    print(revolute_df)
    print(prismatic_df)
    moving_joints_ids = info.moving_joint_info()

    # Determine robot gait type
    if not args.load_config:
        pyb_gaits = Gaits(robot_id)
        print("Gait type: ", args.gait)

        all_joint_pos = []
        reward_list, cum_reward, cum_reward_list = [], 0, []
        # Start simulation
        for i in range(args.timesteps):
            # Initialise gait movement
            if args.gait == "concertina_locomotion":
                pyb_gaits.concertina_locomotion()
            else:
                pyb_gaits.lateral_undulation()
                #pyb_gaits.concertina_locomotion()
            
            # Attach head camera 
            if args.camera == 1:
                pyb_setup.camera()
            
            # If environment = "maze", check if the robot has reached the goal
            pos, ori = info.base_info()
            if args.env == "maze":
                goal_pos = [goal_positions[0][0], goal_positions[1][0]]
                reward = np.linalg.norm(np.array(goal_pos) - np.array(pos)[:2])
                print("Base position: ", pos)
                #calculate reward based on remaining distance to goal
                print("Reward: ", -reward)
                reward_list.append(reward)
                cum_reward += reward
                cum_reward_list.append(cum_reward)

            # Save all past joint positions
            joint_pos, all_joint_pos = info.joint_position(all_joint_pos)

            p.stepSimulation()
            time.sleep(1/240)
    
        # Convert past joint positions to dataframe and export to csv
        all_joint_pos_df = pd.DataFrame(all_joint_pos)
        all_joint_pos_df.to_csv('pyperbot_v2/results/manual/csv/joint_positions.csv', index = False) 
    else:
        # Load joint positions from csv
        try:
            all_joint_pos_df = pd.read_csv(os.path.join(results_dir, 'PPO', 'csv', f'seed{args.seed}_joint_positions.csv'))
            #read joint positions at every line
            for i in range(len(all_joint_pos_df)):
                joint_pos = all_joint_pos_df.iloc[i].values
                joint_pos = joint_pos[1:]
                #apply joint positions to the robot
                for j in range(len(joint_pos)):
                    p.setJointMotorControl2(robot_id, moving_joints_ids[j], p.POSITION_CONTROL, joint_pos[j])
                #step the simulation
                p.stepSimulation()
                time.sleep(1/240)

        except(FileNotFoundError):
            print("File not found. Please ensure the file exists in the correct directory.")
            return
    # Plot timesteps vs reward
    plt.plot(reward_list)
    plt.title('Reward vs Timesteps')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.savefig('pyperbot_v2/results/plots/manual_reward_plot.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='snakebot_sim')
    parser.add_argument('-e', '--env', type=str, default='none', help='Sim env: none, maze, lab.')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the environment.')
    parser.add_argument('-lc', '--load_config', type = bool, default = False, help = 'Load a configuration file for the simulation')
    parser.add_argument('-g', '--gait', type=str, default='lateral_undulation', help='Gaits: lateral undulation, concertina_locomotion')
    parser.add_argument('-ng', '--num_goals', type=int, default=3, help='Number of goals (maze env only).')
    parser.add_argument('-c', '--camera', type=int, default=0, help='Attach head camera: 0, 1.')
    parser.add_argument('-ts', '--timesteps', type=int, default=2400, help='Number of timesteps for the simulation.')
    parser.add_argument('-sr', '--server', type=bool, default=False, help='Running code to read data output directly to server')
    args = parser.parse_args()
    if args.server:
        pi_server = Server(csv_file_path)
        pi_server.debug()

        #Loops through the Fetch-Decode_execute cycle of receiving data from the snake, decoding it, and executing (e.g., running sims to generate the next set of joints, then transmitting them back to the RPi)
        while True:
            pi_server.receive_message()

            if(pi_server.get_message_status() == "UNREAD"):
                pi_server.execute_message()
    else:
        main(args)
    