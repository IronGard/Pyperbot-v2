
import pybullet as p
import os
import time
import pybullet_data
import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from snakebot_sim_argparse import parse_args
from utils.structure_loader import Loader
from utils.gaits import Gaits
from utils.snakebot_info import Info
from utils.server_code import server

# Arguements
args = parse_args()
sim_env = args.env
gait_type = args.gait
num_goals = args.num_goals
timesteps = args.timesteps
camera = args.cam
make_server = args.server

#declarables
csv_file_path = 'pyperbot_v2/results/csv/joint_positions.csv'
transmitting_data = []
host = '0.0.0.0'
port = 6969

def main():
    pyb_setup = Loader("--mp4=results/videos/training_video.mp4")
    pyb_setup.plane()
    
    if sim_env == "lab":
        pyb_setup.lab("pyperbot_v2/snakebot_description/meshes/lab_floor_plan.stl")
    elif sim_env == "maze":
        pyb_setup.maze("pyperbot_v2/snakebot_description/meshes/maze_10x10.stl")
        goal_positions = pyb_setup.goal(num_goals)
    else:
        print('No selected environment.')
    
    robot_id = pyb_setup.robot("pyperbot_v2/snakebot_description/urdf/full_snakebot_no_macro.urdf.xacro")
    moving_joints = [p.getJointInfo(robot_id, joint) for joint in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, joint)[2] != p.JOINT_FIXED]
    moving_joint_ids = [p.getJointInfo(robot_id, joint)[0] for joint in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, joint)[2] != p.JOINT_FIXED]
    print(moving_joints)
    print(moving_joint_ids)
    num_moving_joints = len(moving_joints)
    pyb_gaits = Gaits(robot_id)
    print("Gait type: ", gait_type)
    all_joint_pos = []
    joint_pos = []
    reward_list = []
    cum_reward = 0
    cum_reward_list = []
    for i in range(args.timesteps):
        #initialise gait selection
        if gait_type == "concertina_locomotion":
                pyb_gaits.concertina_locomotion()
        else:
            pyb_gaits.lateral_undulation()
        if camera == 1:
            pyb_setup.camera()
            
        pos, ori = p.getBasePositionAndOrientation(robot_id)
        euler = p.getEulerFromQuaternion(ori)
        #if environment = "maze", check if the robot has reached the goal
        if sim_env == "maze":
            goal_pos = [goal_positions[0][0], goal_positions[1][0]]
            reward = np.linalg.norm(np.array(goal_pos) - np.array(pos)[:2])
            print("Base position: ", pos)
            #calculate reward based on remaining distance to goal
            print("Reward: ", -reward)
            reward_list.append(reward)
            cum_reward += reward
            cum_reward_list.append(cum_reward)
        #save joint positions to csv
        for j in range(len(moving_joint_ids)):
            joint_pos.append(p.getJointState(robot_id, moving_joint_ids[j])[0])
        all_joint_pos.append(joint_pos)
        joint_pos = [] #reset joint_pos array
        p.stepSimulation()
        time.sleep(1/240)
    
    joint_pos = np.array(all_joint_pos)
    df = pd.DataFrame(all_joint_pos)
    df.to_csv('pyperbot_v2/results/csv/joint_positions.csv', index = False) 

    #plot timesteps vs reward
    plt.plot(reward_list)
    plt.title('Reward vs Timesteps')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.savefig('pyperbot_v2/results/plots/manual_reward_plot.png')
    plt.show()

if __name__ == "__main__":
    # main()
    my_server = server(csv_file_path)
    #my_server.debug()

    #Loops through the Fetch-Decode_execute cycle of receiving data from the snake, decoding it, and executing (e.g., running sims to generate the next set of joints, then transmitting them back to the RPi)
    while True:
        my_server.receive_message()

        if(my_server.get_message_status() == "UNREAD"):
            my_server.execute_message()