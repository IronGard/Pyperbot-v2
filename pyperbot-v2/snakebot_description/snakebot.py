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

#Defining the server constants
host = '0.0.0.0'
port = 6969

#import the model environment
import stable_baselines3 as sb3
from stable_baselines3 import PPO, DDPG, A2C, DQN #imports of relevant algorithms (to be fully implemented)
from stable_baselines3.common.env_util import make_vec_env

#TODO: Adjust the values of the joint state publishing and the output rate for the information.

# p.connect(p.GUI)
# p.resetSimulation()
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.81)
# p.setRealTimeSimulation(0)
# plane = p.createCollisionShape(p.GEOM_PLANE)

#setting path to save model and video
print(os.getcwd())#needs ffmpeg installed

#saving and loading video of training
# mp4log = p.startStateLogging(loggingType = 2, fileName = os.path.join(os.getcwd(), 'result', 'videos', 'training_video.mp4'))

# #loading snake robot into the environment
# robot = p.loadURDF("snakebot_description/urdf/updated_snakebot.urdf.xacro", [0, 0, 0], globalScaling = 2.0)
# planeID = p.loadURDF("plane.urdf", [0, 0, 0])

#setup debug camera

#storing joint info in arrays
moving_joint_names = []
moving_joint_inds = []
moving_joint_types = []
moving_joint_limits = []
moving_joint_centers = []

#setting up the joint info
def get_joint_info(robot):
    print('The system has', p.getNumJoints(robot), 'joints')
    for i in range(p.getNumJoints(robot)):
        joint_info  = p.getJointInfo(robot, i)
        if joint_info[2] != (p.JOINT_FIXED):
            moving_joint_names.append(joint_info[1])
            moving_joint_inds.append(joint_info[0])
            moving_joint_types.append(joint_info[2])
            moving_joint_limits.append(joint_info[8:10])
            moving_joint_centers.append((joint_info[8] + joint_info[9])/2)
            print('Joint', i, 'is named', joint_info[1], 'and is of type', joint_info[2])

# get_joint_info(robot)

def rearrange_joint_array(joint_position):
    return_array = np.zeros((4, 5))
    return_array[0] = joint_position[0:5]
    return_array[1] = joint_position[5:10]
    return_array[2] = joint_position[10:15]
    return_array[3] = joint_position[15:20]
    return return_array

def test():
    time.sleep(2)
    while True:
        print("This is running.")
        time.sleep(1)

#breaking joints down into different modules
module_1 = moving_joint_names[0:5]
module_2 = moving_joint_names[5:10]
module_3 = moving_joint_names[10:15]
module_4 = moving_joint_names[15:20]
combined_modules = [module_1, module_2, module_3, module_4]
print(combined_modules)

#break joints into prismatic and revolute joints
prismatic_joints = []

# #set controllers for the moving joints 
#TODO: adjust joints to be controlled by user
num_moving_joints = len(moving_joint_names)
# def joint_position_controller(joint_ind, lower_limit, upper_limit, initial_position):
#     info = p.getJointInfo(robot, joint_ind)
#     joint_params = p.addUserDebugParameter(info[1].decode("utf-8"), lower_limit, upper_limit, initial_position)
#     joint_info = [joint_ind, joint_params]
#     return joint_info
# #setting sidewinding movement parameters for sine wave
dt = 1./240. #Period in pybullet
SNAKE_PERIOD = 0.1 #snake speed
waveLength = 2
wavePeriod = 1.5
waveAmplitude = 0.5
waveFront = 0.0
segmentLength = 0.2
steering = 0.0

base_position_arr = []
base_orientation_arr = []
all_joint_positions = []
all_joint_velocities = []

# # #running simulation
# for i in range(2400):
#     keys = p.getKeyboardEvents()
#     for k, v in keys.items():
#         if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
#             steering = -.2
#         if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED)):
#             steering = 0
#         if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
#             steering = .2
#         if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED)):
#             steering = 0
#     amp = 0.2
#     offset = 0.6
#     numMuscles = p.getNumJoints(robot)
#     scaleStart = 1.0
#     if (waveFront < segmentLength * 4.0):
#         scaleStart = waveFront / (segmentLength * 4.0)
#     segment = numMuscles - 1
#     for joint in range(num_moving_joints):
#         segmentName = moving_joint_names[joint]
#         phase = (waveFront - (segment + 1) * segmentLength) / waveLength
#         phase -= math.floor(phase)
#         phase *= math.pi * 2.0

#         #map phase to curvature
#         targetPos = math.sin(phase) * waveAmplitude
#         p.setJointMotorControl2(robot,
#                             joint,
#                             p.POSITION_CONTROL,
#                             targetPosition=targetPos + steering,
#                             force=30)
#         #moving the joint by squashing sine wave
#     waveFront += dt/wavePeriod*waveLength

#     #getting position and the orientation of the robot
#     pos, ori = p.getBasePositionAndOrientation(robot)
#     euler = p.getEulerFromQuaternion(ori)
#     base_position_arr.append(pos)
#     base_orientation_arr.append(ori)
#     p.stepSimulation()
#     joint_states = p.getJointStates(robot, moving_joint_inds)
#     joint_positions = [state[0] for state in joint_states]
#     joint_velocities = [state[1] for state in joint_states]
#     joint_torques = [state[3] for state in joint_states]
#     all_joint_positions.append(joint_positions)
#     all_joint_velocities.append(joint_velocities)
#     print('---------------------')
#     print('Joint Positions:', joint_positions)
#     print('Joint Velocities:', joint_velocities)
#     print('Joint Torques:', joint_torques)
#     print('---------------------')
#     time.sleep(dt)

# #generating plots for position and orientation w.r.t time
# fig, axs = plt.subplots(2)
# fig.suptitle('Position and Orientation of the Snakebot')
# axs[0].plot(base_position_arr)
# axs[1].plot(base_orientation_arr)
# plt.savefig(os.path.join(os.getcwd(), 'results', 'plots', 'position_orientation_plot.png'))
# plt.show()

# #save joint_positions and joint_velocities to csv files
# joint_positions_df = pd.DataFrame(all_joint_positions)
# joint_velocities_df = pd.DataFrame(all_joint_velocities)
# joint_positions_df.to_csv(os.path.join(os.getcwd(), 'results', 'csv', 'joint_positions.csv'))
# joint_velocities_df.to_csv(os.path.join(os.getcwd(), 'results', 'csv', 'joint_velocities.csv'))

#Getting joint values frm joint_positions.csv
joint_values = np.zeros((4,5))
csv_file_path = os.path.join(os.getcwd(), 'results', 'csv', 'joint_positions.csv')
sample_step = 20
counter = 0
starting_sample = 150

# def rearrange_array(old_array):
#     joint_values[0] = old_array[0:5]
#     joint_values[1] = old_array[5:10]
#     joint_values[2] = old_array[10:15]
#     joint_values[3] = old_array[15:20]
#     return joint_values
#Setting up the server
def setup_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket has been created")

    try:
        s.bind((host, port))
    
    except socket.error as msg:
        print(msg)
    
    print("Socket bind complete.")

    return s

def setup_connection():
    s.listen(1)
    conn, address = s.accept()
    print("Connected to: " +address[0]+ ": " +str(address[1]))
    return conn

def data_transfer(conn, transmitting_data):
    while True:
        #Receive data from the snake.
        data = conn.recv(1024)
        data = data.decode('utf-8')

        dataMessage = data.split(' ', 1)
        command = dataMessage[0]

        if(command == 'REQ'):
            reply = transmitting_data
        
        elif(command == 'EXIT'):
            print("Client has left.")
            break

        elif(command == 'KILL'):
            print("Our server is shutting down.")
            s.close()
            break
        else:
            reply = "Unknown command."
        
        conn.sendall(str.encode(reply))
    conn.close()

s = setup_server()
while True:
    try:
        conn = setup_connection()
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                if counter >= starting_sample:
                   if counter % sample_step == 0:
                        transmitting_data = str(line)
                        data_transfer(conn, transmitting_data)
                counter += 1 #Setting up the server    
    except:
        break

#closing simulation
p.disconnect()