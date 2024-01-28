import pybullet as p
import os
import time
import math
import pybullet_data

p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(0.0001)

#setting path from the information


robot = p.loadURDF(os.path.join())

def get_joint_info(robot):
    print('The system has', p.getNumJoints(robot), 'joints')