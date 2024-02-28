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
        # arrow up: 65297, arrow down: 65298, arrow right: 65296, arrow left: 65295
        input_dict = {65297:  1,
                      65298: -1,
                      65296:  1j,
                      65295: -1j}
        pressed_keys = []
        direction = 0
        events = p.getKeyboardEvents()
        key_codes = events.keys()
        for key in key_codes:
            print(key)
            if key in input_dict:
                pressed_keys.append(key)
                direction += input_dict[key]
        return pressed_keys, direction

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

joint_classification(robot)

anisotropicFriction = [1, 0.01, 0.01]
lateralFriction = 2
p.changeDynamics(robot, -1, lateralFriction = lateralFriction, anisotropicFriction=anisotropicFriction)
for i in range(p.getNumJoints(robot)):
    p.changeDynamics(robot, i, lateralFriction = lateralFriction,anisotropicFriction=anisotropicFriction)

dt = 1./240. #Period in pybullet
waveLength = 2 #Wave length of the snake
wavePeriod = 1 #Wave period of the snake
waveAmplitude = 0.5 #Amplitude of the snake
waveFront = 0 #Front of the wave of the snake
steering = 0 #Steering of the snake
segmentLength = 0.4 #Length of the snake

while True:
    scaleStart = 1.0
    steering = 0

    if (waveFront < segmentLength * 4.0):
        scaleStart = waveFront / (segmentLength * 4.0)
    segment = 4 - 1
    joint_list = [6,  14,  22,  30]
    for joint in range(4):
        segmentName = p.getJointInfo(robot, joint_list[joint])
        segment = joint
        phase = (waveFront - (segment + 1) * segmentLength) / waveLength
        phase -= math.floor(phase)
        phase *= math.pi * 2.0

        #map phase to curvature
        targetPos = math.sin(phase) * scaleStart* waveAmplitude
        p.setJointMotorControl2(robot,
                            joint_list[joint],
                            p.POSITION_CONTROL,
                            targetPosition=targetPos + steering,
                            force=30)
        #moving the joint by squashing sine wave

    waveFront += dt/wavePeriod*waveLength
    p.stepSimulation()

    time.sleep(dt)
    

