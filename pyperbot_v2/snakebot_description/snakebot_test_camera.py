import pybullet as p
import os
import numpy as np
import sys

from snakebot_argparse import parse_args
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from utils.structure_loader import Loader

args = parse_args()
print(args)

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
print("Numpy acceleration: ", p.isNumpyEnabled())
pyb_setup = Loader("--mp4=results/videos/training_video.mp4")
pyb_setup.plane()
pyb_setup.maze("pyperbot_v2/snakebot_description/meshes/maze_10x10.stl")
#p.loadURDF("pyperbot_v2/snakebot_description/urdf/full_snakebot_no_macro.urdf.xacro")
robot = pyb_setup.robot("pyperbot_v2/snakebot_description/urdf/full_snakebot_no_macro.urdf.xacro")
pyb_setup.goal(num_goals)

fov, aspect, nearplane, farplane = 100, 1.0, 0.01, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)


def camera():
    # World position (xyz) and orientation (xyzw) of link 0 (head) 
    pos, ori, _, _, _, _ = p.getLinkState(robot, 0, computeForwardKinematics=True)
    # generate a list of 9 floats from quaternion (xyzw)
    rot_mat = p.getMatrixFromQuaternion(ori)
    # reshape the list of 9 floats into 3x3 transformation matrix
    rot_mat = np.array(rot_mat).reshape(3, 3)
    # camera target position and up vector (xyz)
    cam_target = (-1, 0, 0)
    up_vector = (0, 0, 1)
    # transfrom target position and up vector 
    camera_vector = rot_mat.dot(cam_target)
    up_vector = rot_mat.dot(up_vector)
    # generate the view matrix
    view_matrix = p.computeViewMatrix(pos, pos + 0.1 * camera_vector, up_vector)
    img = p.getCameraImage(100, 100, view_matrix, projection_matrix)
    return img

print(p.isNumpyEnabled())
# Main loop
while True:
    p.stepSimulation()
    camera()