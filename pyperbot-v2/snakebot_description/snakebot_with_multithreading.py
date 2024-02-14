import pybullet as p
import os
import time
import math
import pybullet_data
import threading
import matplotlib.pyplot as plt

#storing joint info in arrays
moving_joint_names = []
moving_joint_inds = []
moving_joint_types = []
moving_joint_limits = []
moving_joint_centers = []

# #setting sidewinding movement parameters for sine wave
dt = 1./240. #Period in pybullet
SNAKE_PERIOD = 0.1 #snake speed
wavePeriod = SNAKE_PERIOD
waveLength = 2
wavePeriod = 1.5
waveAmplitude = 0.5
waveFront = 0.0
segmentLength = 0.2
steering = 0.0

#setting up the joint info
def get_joint_info(robot):
    '''
    Preliminary function to return relevant joint information pertaining to the joints of the robot.
    '''
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

def joint_position_controller(robot, joint_ind, lower_limit, upper_limit, initial_position):
    '''
    Function used to control the joint positions based on the limits of the robot.
    '''
    info = p.getJointInfo(robot, joint_ind)
    joint_params = p.addUserDebugParameter(info[1].decode("utf-8"), lower_limit, upper_limit, initial_position)
    joint_info = [joint_ind, joint_params]
    return joint_info


def sample(joint_positions, joint_velocities, joint_torques):
    '''
    Function to sample joint positions and velocities every second
    '''
    while True:
        print('---------------------')
        print('Joint Positions:', joint_positions)
        print('Joint Velocities:', joint_velocities)
        print('Joint Torques:', joint_torques)
        print('---------------------')
        time.sleep(1)


def test():
    time.sleep(2)
    while True:
        print("This is running.")
        time.sleep(1)

# #running simulation

def run_simulation(robot, waveFront):
    '''
    Function to run the simulation of the snakebot.
    '''
    #TODO: Adjust the values of the joint state publishing and the output rate for the information.
    

    # #set controllers for the moving joints 
    #TODO: adjust joints to be controlled by user
    num_moving_joints = len(moving_joint_names)
    while True:
        amp = 0.2
        offset = 0.6
        numMuscles = p.getNumJoints(robot)
        scaleStart = 1.0
        if (waveFront < segmentLength * 4.0):
            scaleStart = waveFront / (segmentLength * 4.0)
        segment = numMuscles - 1
        for joint in range(num_moving_joints):
            segmentName = moving_joint_names[joint]
            phase = (waveFront - (segment + 1) * segmentLength) / waveLength
            phase -= math.floor(phase)
            phase *= math.pi * 2.0

            #map phase to curvature
            targetPos = math.sin(phase) * waveAmplitude
            p.setJointMotorControl2(robot,
                                joint,
                                p.POSITION_CONTROL,
                                targetPosition=targetPos,
                                force=30)
            #moving the joint by squashing sine wave
        waveFront += dt/wavePeriod*waveLength

        #getting position and the orientation of the robot
        pos, ori = p.getBasePositionAndOrientation(robot)
        euler = p.getEulerFromQuaternion(ori)
        # print('Position:', pos)
        # print('Orientation:', ori)
        # print('Euler:', euler)
        p.stepSimulation()
        joint_states = p.getJointStates(robot, moving_joint_inds)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        time.sleep(dt)


#start threads
p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)
plane = p.createCollisionShape(p.GEOM_PLANE)

#setting path to save model and video
print(os.getcwd())#needs ffmpeg installed

#loading snake robot into the environment
robot = p.loadURDF("snakebot_description/urdf/updated_snakebot.urdf.xacro", [0, 0, 0], globalScaling = 2.0)
planeID = p.loadURDF("plane.urdf", [0, 0, 0])

get_joint_info(robot)

module_1 = moving_joint_names[0:5]
module_2 = moving_joint_names[5:10]
module_3 = moving_joint_names[10:15]
module_4 = moving_joint_names[15:20]
combined_modules = [module_1, module_2, module_3, module_4]

#setting up threads
thread_one = threading.Thread(target = run_simulation, args = (robot, waveFront))
thread_two = threading.Thread(target = test)

thread_one.start()
thread_two.start()


#closing simulation
p.disconnect()