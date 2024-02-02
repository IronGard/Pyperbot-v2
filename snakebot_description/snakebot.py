import pybullet as p
import os
import time
import math
import pybullet_data

p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)
plane = p.createCollisionShape(p.GEOM_PLANE)

#setting path to save model and video
print(os.getcwd())#needs ffmpeg installed

#saving and loading video of training
mp4log = p.startStateLogging(loggingType = 2, fileName = os.path.join(os.getcwd(), 'result', 'videos', 'training_video.mp4'))

#loading snake robot into the environment
robot = p.loadURDF("snakebot_description/urdf/updated_snakebot.urdf.xacro", [0, 0, 0])
planeID = p.loadURDF("plane.urdf", [0, 0, 0])

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

get_joint_info(robot)

#breaking joints down into different modules
module_1 = moving_joint_names[0:5]
module_2 = moving_joint_names[5:10]
module_3 = moving_joint_names[10:15]
module_4 = moving_joint_names[15:20]
combined_modules = [module_1, module_2, module_3, module_4]
print(combined_modules)

# #set controllers for the moving joints
num_moving_joints = len(moving_joint_names)
def joint_position_controller(joint_ind, lower_limit, upper_limit, initial_position):
    info = p.getJointInfo(robot, joint_ind)
    joint_params = p.addUserDebugParameter(info[1].decode("utf-8"), lower_limit, upper_limit, initial_position)
    joint_info = [joint_ind, joint_params]
    return joint_info
# #setting sidewinding movement parameters for sine wave
dt = 1./240.
SNAKE_PERIOD = 0.1
wavePeriod = SNAKE_PERIOD
waveLength = 4
wavePeriod = 1.5
waveAmplitude = 0.5
waveFront = 0.0
segmentLength = 0.2
steering = 0.0


# #running simulation
while(1):
    keys = p.getKeyboardEvents()
    for k, v in keys.items():
        if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
            steering = -.2
        if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED)):
            steering = 0
        if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
            steering = .2
        if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED)):
            steering = 0
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
                            targetPosition=targetPos + steering,
                            force=30)
        #moving the joint by squashing sine wave
    waveFront += dt/wavePeriod*waveLength
    p.stepSimulation()
    joint_states = p.getJointStates(robot, moving_joint_inds)
    print(joint_states)
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    
    
    print('---------------------')
    print('Joint Positions:', joint_positions)
    print('Joint Velocities:', joint_velocities)
    print('Joint Torques:', joint_torques)
    print('---------------------')
    time.sleep(dt)


#closing simulation
p.disconnect()
p.stopStateLogging(mp4log)