import pybullet as p
import os
import time
import math
import pybullet_data
import threading
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import socket
import csv
import sys
import random

#import the model environment
import stable_baselines3 as sb3
from stable_baselines3 import PPO, DDPG, A2C, DQN #imports of relevant algorithms (to be fully implemented) 
from stable_baselines3.common.env_util import make_vec_env

#establish relative import pipeline
# resources_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(resources_dir, '..'))
# from resources.goal import Human, Duck

from snakebot_argparse import parse_args
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from utils.structure_loader import Loader

args = parse_args()
print(args)
#====================================CONSTANTS====================================
#Defining the server constants
host = '0.0.0.0'
port = 6969

#Defining snake movement/sinusoidal movement parameters
dt = args.delta_time #Period in pybullet
SNAKE_PERIOD = args.snake_period #Period of the snake
waveLength = args.waveLength #Wave length of the snake
wavePeriod = args.wavePeriod #Wave period of the snake
waveAmplitude = args.waveAmplitude #Amplitude of the snake
waveFront = args.waveFront #Front of the wave of the snake
segmentLength = args.segmentLength #Length of the snake
steering = args.steering #Steering of the snake
amp = args.amp #Amplitude of the snake
offset = args.offset #Offset of the snake
num_goals = args.num_goals #Number of goals
#================================================================================

pyb_setup = Loader("--mp4=results/videos/training_video.mp4")
pyb_setup.plane()
pyb_setup.maze("pyperbot_v2/snakebot_description/meshes/maze_10x10.stl")
robot = pyb_setup.robot("pyperbot_v2/snakebot_description/urdf/test_snakebot.urdf.xacro")
pyb_setup.goal(num_goals)

fov, aspect, nearplane, farplane = 100, 1.0, 0.01, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)

def camera():
    # Center of mass position and orientation (of link-7)
    com_p, com_o, _, _, _, _ = p.getLinkState(robot, 0, computeForwardKinematics=True)
    rot_matrix = p.getMatrixFromQuaternion(com_o)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # Initial vectors
    init_camera_vector = (0, 0, 1) # z-axis
    init_up_vector = (0, 1, 0) # y-axis
    # Rotated vectors
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)
    view_matrix = p.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
    img = p.getCameraImage(1000, 1000, view_matrix, projection_matrix)
    return img

# Main loop
while True:
    p.stepSimulation()
    camera()