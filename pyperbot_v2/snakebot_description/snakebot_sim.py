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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from snakebot_sim_argparse import parse_args
from utils.structure_loader import Loader
from utils.gaits import Gaits
from utils.snakebot_info import Info
from utils.server_code import Server

# Arguments
args = parse_args()
sim_env = args.env
gait_type = args.gait
num_goals = args.num_goals
timesteps = args.timesteps
camera = args.cam
make_server = args.server

abs_path = os.path.abspath(__file__)


# Declarables
csv_file_path = 'pyperbot_v2/results/csv/joint_positions.csv'
transmitting_data = []
host = '0.0.0.0'
port = 6969

def main():
    pyb_setup = Loader("--mp4=results/videos/training_video.mp4")

    # Setup base plane
    pyb_setup.plane()
    
    # Setup simulation environment
    if sim_env == "lab":
        pyb_setup.lab("pyperbot_v2/snakebot_description/meshes/lab_floor_plan.stl")
    elif sim_env == "maze":
        pyb_setup.maze("pyperbot_v2/snakebot_description/meshes/maze_10x10.stl")
        goal_positions = pyb_setup.goal(num_goals)
    else:
        print('No selected environment.')
    
    # Setup robot
    robot_id = pyb_setup.robot("pyperbot_v2/snakebot_description/urdf/updated_full_snakebot.urdf.xacro")

    # Obtain robot information
    info = Info(robot_id)
    revolute_df, prismatic_df, _, _ = info.joint_info()
    moving_joints_ids = info.moving_joint_info()

    # Determine robot gait type
    pyb_gaits = Gaits(robot_id)
    print("Gait type: ", gait_type)

    all_joint_pos = []
    reward_list, cum_reward, cum_reward_list = [], 0, []
    # Start simulation
    for i in range(args.timesteps):
        # Initialise gait movement
        if gait_type == "concertina_locomotion":
            pyb_gaits.concertina_locomotion()
        else:
            pyb_gaits.lateral_undulation()

        # Attach head camera 
        if camera == 1:
            pyb_setup.camera()
        
        # If environment = "maze", check if the robot has reached the goal
        pos, ori = info.base_info()
        if sim_env == "maze":
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
    all_joint_pos_df.to_csv('pyperbot_v2/results/csv/joint_positions.csv', index = False) 

    # Plot timesteps vs reward
    plt.plot(reward_list)
    plt.title('Reward vs Timesteps')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.savefig('pyperbot_v2/results/plots/manual_reward_plot.png')
    plt.show()

if __name__ == "__main__":
    if make_server:
        pi_server = Server(csv_file_path)
        pi_server.debug()

        #Loops through the Fetch-Decode_execute cycle of receiving data from the snake, decoding it, and executing (e.g., running sims to generate the next set of joints, then transmitting them back to the RPi)
        while True:
            pi_server.receive_message()

            if(pi_server.get_message_status() == "UNREAD"):
                pi_server.execute_message()
    else:
        main()
    