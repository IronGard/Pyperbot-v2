#Reads and instantiates the human.urdf file
import pybullet as p
import pybullet_data
import os

#get the human.urdf file from the pybullet_data path
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

class Human:
    def __init__(self, client, basePosition = [0, 0, 0]):
        '''
        Constructor for the Human class to define a human object (used as the "goal" in the environment)
        '''
        self.client = client
        p.loadURDF("humanoid.urdf", basePosition, globalScaling = 1.0, physicsClientId = client)
        #taken from the human.urdf file in the pybullet_data path

class Duck:
    def __init__(self, client, basePosition = [0, 0, 0]):
        '''
        Constructor for the Duck class to define a duck object (used as the "goal" in the environment)
        '''
        self.client = client
        p.loadURDF("duck_vhacd.urdf", basePosition, globalScaling = 1.0, physicsClientId = client)