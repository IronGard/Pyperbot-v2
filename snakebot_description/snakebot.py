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
print(os.getcwd())
os.path.join(os.getcwd(), 'models')
os.path.join(os.getcwd(), 'videos') #needs ffmpeg installed

#saving and loading video of training
mp4log = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, os.path.join(os.getcwd(), 'videos', 'training_video.mp4'))

#loading snake robot into the environment
robot = p.loadURDF("snakebot_description/urdf/snakebot.urdf.xacro", [0, 0, 0], useFixedBase=True)

#initialising collisions between each joint


#setting up the joint info
def get_joint_info(robot):
    print('The system has', p.getNumJoints(robot), 'joints')
    for i in range(p.getNumJoints(robot)):
        print(p.getJointInfo(robot, i))

get_joint_info(robot)

#importing the snake movement

#setting sidewinding movement
dt = 1./240.
SNAKE_PERIOD = 0.1
wavePeriod = SNAKE_PERIOD
waveLength = 4
wavePeriod = 1.5
waveAmplitude = 0.5
waveFront = 0.0
segmentLength = 0.2
steering = 0.0


#running simulation
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
    for joint in range(p.getNumJoints(robot)):
        segmentName = joint
        phase = (waveFront - (segmentName + 1) * segmentLength) / waveLength
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
    time.sleep(dt)
p.disconnect()
p.stopStateLogging(mp4log)