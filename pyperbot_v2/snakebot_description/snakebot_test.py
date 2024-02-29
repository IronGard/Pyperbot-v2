import pybullet as p
import os
import numpy as np
import sys
import time
import math

from snakebot_argparse import parse_args
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from utils.structure_loader import Loader

args = parse_args()
print(args)
num_goals = args.num_goals #Number of goals
#================================================================================
print("Numpy acceleration: ", p.isNumpyEnabled())
pyb_setup = Loader("--mp4=results/videos/training_video.mp4")
pyb_setup.plane()
#pyb_setup.maze("pyperbot_v2/snakebot_description/meshes/maze_10x10.stl")
#p.loadURDF("pyperbot_v2/snakebot_description/urdf/full_snakebot_no_macro.urdf.xacro")
robot = pyb_setup.robot("pyperbot_v2/snakebot_description/urdf/updated_full_snakebot_no_macro.urdf.xacro")
#pyb_setup.goal(num_goals)
print(p.getNumJoints(robot))
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

def get_keyboard_input():
    # Map direction to steering. Up: 65297, Down: 65298, Right: 65296, Left: 65295
    input_dict = {65297: 1j, 65298: -1j, 65296:  1, 65295: -1}
    steering_dict = {1: 0.8, -1: -0.8, 1j: 0, -1j: 0, 1+1j: 0.4, -1+1j : -0.4}

    pressed_keys, direction = [], 0
    key_codes = p.getKeyboardEvents().keys()
    for key in key_codes:
        if key in input_dict:
            pressed_keys.append(key)
            direction += input_dict[key]

    steering = steering_dict.get(direction, 0)

    return steering

def joint_classification(robot):
    revolute_joints = []
    prismatic_joints = []
    fixed_joints = []
    other_joints = []
    for i in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, i)
        if joint_info[2] == 0:
            revolute_joints.append(i)
        elif joint_info[2] == 1:
            prismatic_joints.append(i)
        elif joint_info[2] == 4:
            fixed_joints.append(i)
        else:
            other_joints.append(i)
    print("revolute joints: ", revolute_joints)
    print("prismatic joints: ", prismatic_joints)
    print("fixed joints: ", fixed_joints)
    print("other joints", other_joints)

def lateral_undulation(waveFront):
    # Physics setting
    anisotropicFriction = [0.005, 0.005, 1]
    lateralFriction = 2
    for i in range(-1, p.getNumJoints(robot)):
        p.changeDynamics(robot, i, lateralFriction = lateralFriction,anisotropicFriction=anisotropicFriction)

    waveLength = 4.2 #Wave length of the snake
    wavePeriod = 1 #Wave period of the snake
    dt = 1./240. #Period in pybullet
    waveAmplitude = 1 #Amplitude of the snake
    segmentLength = 0.4 #Length of the snake

    scaleStart = 1.0
    if (waveFront < segmentLength * 4.0):
        scaleStart = waveFront / (segmentLength * 4.0)

    joint_list = [2, 6, 10, 14, 18, 22, 26, 30]
    steering = get_keyboard_input()

    for joint in range(p.getNumJoints(robot)):
        segment = joint
        phase = (waveFront - (segment + 1) * segmentLength) / waveLength
        phase -= math.floor(phase)
        phase *= math.pi * 2.0

        #map phase to curvature
        targetPos = math.sin(phase)* scaleStart* waveAmplitude + steering
        if joint not in joint_list:
            targetPos = 0
        p.setJointMotorControl2(robot, joint, p.POSITION_CONTROL, targetPosition=targetPos, force=30)

    waveFront += dt/wavePeriod*waveLength
    return waveFront

waveFront = 0
while True:
    waveFront = lateral_undulation(waveFront)
    p.stepSimulation()
    time.sleep(1./240.)

'''
# Physics setting
anisotropicFriction = [0.005, 0.005, 1]
lateralFriction = 2
for i in range(-1, p.getNumJoints(robot)):
    p.changeDynamics(robot, i, lateralFriction = lateralFriction,anisotropicFriction=anisotropicFriction)

# Parameters
waveLength = 4.2 #Wave length of the snake
wavePeriod = 1 #Wave period of the snake
dt = 1./240. #Period in pybullet
waveAmplitude = 1 #Amplitude of the snake
waveFront = 0 #Front of the wave of the snake
steering = 0 #Steering of the snake
segmentLength = 0.4 #Length of the snake
scaleStart = 1.0

while True:
    waveFront = lateral_undulation(waveFront)
    
    scaleStart = 1
    if (waveFront < segmentLength * 4.0):
        scaleStart = waveFront / (segmentLength * 4.0)
    steering = get_keyboard_input()
    #segment = 8 - 1
    joint_list = [2, 6,  10, 14,  18, 22,  26, 30]
    for joint in range(33):
        segmentName = p.getJointInfo(robot, joint)
        segment = joint
        phase = (waveFront - (segment + 1) * segmentLength) / waveLength
        phase -= math.floor(phase)
        phase *= math.pi * 2.0

        #map phase to curvature
        targetPos = math.sin(phase) * scaleStart* waveAmplitude
        if joint in joint_list:
            p.setJointMotorControl2(robot,
                                joint,
                                p.POSITION_CONTROL,
                                targetPosition=targetPos + steering,
                                force=30)
        else:
            p.setJointMotorControl2(robot,
                                joint,
                                p.POSITION_CONTROL,
                                targetPosition=0,
                                force=30)
        #moving the joint by squashing sine wave

    waveFront += dt/wavePeriod*waveLength
    
    p.stepSimulation()

    time.sleep(1./240.)
'''

