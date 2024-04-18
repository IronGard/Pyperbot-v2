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
from Pi_Code_folder.server import Server

#Directories
results_dir = "pyperbot_v2/results/"

#TODO: ADD SAMPLING DELAY


# Declarables
csv_file_path = 'pyperbot_v2/results/csv/joint_positions.csv'
transmitting_data = []
host = '0.0.0.0'
port = 6969

def unnormalise_actions(actions, joint_limits):
    # Unnormalise actions
    unnormalised_actions = []
    for i in range(len(actions)):
        unnormalised_actions.append(actions[i] * (joint_limits[i][1] - joint_limits[i][0]) / 2 + (joint_limits[i][1] + joint_limits[i][0]) / 2)
    return unnormalised_actions

def main(args, server):
    pyb_setup = Loader("--mp4=results/videos/training_video.mp4")

    # Setup base plane
    pyb_setup.plane()
    
    # Setup simulation environment
    if args.env == "lab":
        pyb_setup.lab("pyperbot_v2/snakebot_description/meshes/lab_floor_plan.stl")
    elif args.env == "maze":
        pyb_setup.maze("pyperbot_v2/snakebot_description/meshes/maze.stl")
        goal_positions = pyb_setup.goal(num_goals=args.num_goals)
    elif args.env == "small_maze":
        pyb_setup.maze("pyperbot_v2/snakebot_description/meshes/maze_small.stl", basePos = [-5, 0, 0])
        goal_positions = pyb_setup.goal(num_goals=args.num_goals, size=4)
    elif args.env == "terrain":
        pyb_setup.terrain("pyperbot_v2/snakebot_description/meshes/terrain.stl")
    else:
        print('No selected environment.')
        # if args.load_config:
        #     goal_config = configparser.ConfigParser()
        #     goal_config.read(f'pyperbot_v2/config/seeded_run_configs/goal_config/seed{args.seed}_config.ini')
        #     sections = goal_config.sections()
        #     print(sections)
        #     goal_positions = [[int(x) for x in goal_config['GOALS']['goal1'].split(',')], 
        #                       [int(x) for x in goal_config['GOALS']['goal2'].split(',')], 
        #                       [int(x) for x in goal_config['GOALS']['goal3'].split(',')]]
        #     print(goal_positions)
        #     pyb_setup.manual_goal(len(goal_positions), goal_positions)
    # Setup robot
    # if args.load_config:
    #     snake_config = configparser.ConfigParser()
    #     snake_config.read(f'pyperbot_v2/config/seeded_run_configs/snake_config/seed{args.seed}_config.ini')
    #     robot_id = pyb_setup.robot("pyperbot_v2/snakebot_description/urdf/snakebot.urdf.xacro",
    #                                 basePosition = [float(x) for x in snake_config['SNAKEBOT']['basePosition'].split(',')],
    #                                 baseOrientation = [float(x) for x in snake_config['SNAKEBOT']['baseOrientation'].split(',')]
    #                                 )
    # else:
    robot_id = pyb_setup.robot("pyperbot_v2/snakebot_description/urdf/snakebot.urdf.xacro",
                                basePosition = [0, 0, 1],
                                baseOrientation = [0.5, 0, 0, 1]
                                )
    # Obtain robot information
    info = Info(robot_id)
    revolute_df, prismatic_df, _, _ = info.joint_info()
    print(revolute_df)
    print(prismatic_df)
    moving_joints_ids = info.moving_joint_info()

    reward_list = []
    cum_reward_list = []
    cum_reward = 0
    # Determine robot gait type
    if not args.load_config:
        pyb_gaits = Gaits(robot_id)
        print("Gait type: ", args.gait)

        all_joint_pos = []
        reward_list, cum_reward, cum_reward_list = [], 0, []

        # Start simulation
        delay = 200
        counter = 0
        for i in range(args.timesteps):
            
            # Initialise gait movement
            if args.gait == "rectilinear_locomotion":
                pyb_gaits.rectilinear_locomotion()
            elif args.gait == "rectilinear_locomotion_irl":
                pyb_gaits.rectilinear_locomotion_irl()
            else:
                pyb_gaits.lateral_undulation()
            
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
            #checking manual control for standard environment
            if args.env == "none":
                goal_pos = [-5, 0, 0]
                goal_vis = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=5, rgbaColor=[1, 0, 0, 1], visualFramePosition=goal_pos)
                initial_reward = 0
                reward = np.linalg.norm(np.array(goal_pos)[:2] - np.array(pos)[:2])
                print("Base position: ", pos)
                #calculate reward based on remaining distance to goal
                # print("Reward: ", -reward)
                print("Distance to goal: ", reward)
                reward_list.append(-reward)
                cum_reward -= reward
                print("Cumulative reward: ", cum_reward)
                cum_reward_list.append(cum_reward)

            # Save all past joint positions
            joint_pos, all_joint_pos = info.joint_position(all_joint_pos)
            new_joint_pos = ""

            for j in range(len(joint_pos) - 1):
                new_joint_pos = new_joint_pos + str(joint_pos[j]) + ","
            
            new_joint_pos = new_joint_pos + str(joint_pos[len(joint_pos)-1])
            #print("Gene is trolling meeeeee")
            if args.server:
                if(i > delay):
                    print("We have entered this condition")
                    server.set_transmitting_data(new_joint_pos)
                    server.receive_message()

                    if(server.get_message_status() == "UNREAD"):
                        server.execute_message()
                
            p.stepSimulation()
            time.sleep(1/240)
            counter += 1

        # Convert past joint positions to dataframe and export to csv
        all_joint_pos_df = pd.DataFrame(all_joint_pos)
        os.makedirs('pyperbot_v2/results/manual/csv', exist_ok=True)
        all_joint_pos_df.to_csv('pyperbot_v2/results/manual/csv/joint_positions.csv', index = False)

        #save all rewards, cumulative rewards, and base positions
        with open('pyperbot_v2/results/manual/csv/rewards.csv', mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(reward_list)
            writer.writerow(cum_reward_list)
            writer.writerow(pos) 
    else:
        # Load joint positions from csv
        try:
            file_path = input("Enter the trial number to load: ")
            all_joint_pos_df = pd.read_csv(f'pyperbot_v2/logs/actions/episode_{file_path}.csv')
            # all_joint_pos_df = pd.read_csv(os.path.join(results_dir, 'PPO', f'actions_PPO_snakebot_seed{int(args.seed)}.csv'))
            # old_joint_pos_df = pd.read_csv(os.path.join(results_dir, 'PPO', 'csv', f'seed{args.seed}_joint_positions.csv'))
            moving_joint_ids = [2, 3, 8, 11, 12, 17, 20, 21, 26, 29, 30, 35]
            all_moving_joint_ids = [2, 3, 4, 7, 8, 11, 12, 13, 16, 17, 20, 21, 22, 25, 26, 29, 30, 31, 34, 35]
            #read joint positions at every line
            if args.mode == 'position':
                for i in range(len(all_joint_pos_df)):
                    joint_pos = all_joint_pos_df.iloc[i].values
                    joint_pos = unnormalise_actions(joint_pos, info.get_joint_limits())
                    #apply joint positions to the robot\
                    print("Joint positions: ", joint_pos)
                    print("Base Position: ", info.base_info()[0])
                    print("Base Orientation: ", info.base_info()[1])
                    for j in range(len(all_moving_joint_ids)):
                        p.setJointMotorControl2(robot_id, all_moving_joint_ids[j], p.POSITION_CONTROL, joint_pos[j])
                    #step the simulation
                    p.stepSimulation()
                    time.sleep(1/240)
            elif args.mode == 'velocity':
                for i in range(len(all_joint_pos_df)):
                    joint_pos = all_joint_pos_df.iloc[i].values
                    #apply joint positions to the robot\
                    print("Joint positions: ", joint_pos)
                    for j in range(len(all_moving_joint_ids)):
                        p.setJointMotorControl2(robot_id, all_moving_joint_ids[j], p.VELOCITY_CONTROL, joint_pos[j])
                    #step the simulation
                    p.stepSimulation()
                    time.sleep(1/240)
            # elif args.mode == 'torque':
            #     for i in range(len(all_joint_pos_df)):
            #         joint_pos = all_joint_pos_df.iloc[i].values
            #         #apply joint positions to the robot\
            #         print("Joint positions: ", joint_pos)
            #         for j in range(len(all_moving_joint_ids)):
            #             p.setJointMotorControl2(robot_id, all_moving_joint_ids[j], p.TORQUE_CONTROL, joint_pos[j])
            #         #step the simulation
            #         p.stepSimulation()
            #         time.sleep(1/240)

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
    parser.add_argument('-m', '--mode', type=str, default='position', help='Control mode: position, torque, velocity.')
    parser.add_argument('-ng', '--num_goals', type=int, default=3, help='Number of goals (maze env only).')
    parser.add_argument('-c', '--camera', type=int, default=0, help='Attach head camera: 0, 1.')
    parser.add_argument('-ts', '--timesteps', type=int, default=5000, help='Number of timesteps for the simulation.')
    parser.add_argument('-sr', '--server', type=bool, default=False, help='Running code to read data output directly to server')
    args = parser.parse_args()
    if args.server:
        pi_server = Server(csv_file_path)
        pi_server.debug()
        main(args, pi_server)
        # #Loops through the Fetch-Decode_execute cycle of receiving data from the snake, decoding it, and executing (e.g., running sims to generate the next set of joints, then transmitting them back to the RPi)
        # while True:
        #     feed_data = #enter live data
        #     pi_server.set_transmitting_data(feed_data)
        #     pi_server.receive_message()

        #     if(pi_server.get_message_status() == "UNREAD"):
        #         pi_server.execute_message()
    else:
        main(args, server = None)
    