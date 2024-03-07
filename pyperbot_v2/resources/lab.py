import pybullet as p
import pybullet_data
import os

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

class Lab:
    def __init__(self, client):
        self.client = client
        self.lab = p.loadURDF("pyperbot_v2/resources/lab.urdf",
                              [0, 0, 0],
                              globalScaling = 1.0, 
                              physicsClientId = self.client)
        