#Reads and instantiates the human.urdf file
import pybullet as p
import pybullet_data
import random
import os


class Human:
    def __init__(self, client, basePosition = [0, 0, 0]):
        '''
        Constructor for the Human class to define a human object (used as the "goal" in the environment)
        '''
        self._client = client
        self._human = self._client.loadURDF("humanoid.urdf", 
                                            basePosition, 
                                            globalScaling = 1.0)
        #taken from the human.urdf file in the pybullet_data path

class Duck:
    def __init__(self, client, basePosition = [0, 0, 0]):
        '''
        Constructor for the Duck class to define a duck object (used as the "goal" in the environment)
        '''
        self._client = client
        self._duck = self._client.loadURDF("duck_vhacd.urdf", 
                                           basePosition, 
                                           globalScaling = 1.0)

class Goal:
    def __init__(self, client, num_goals):
        '''
        Constructor for the Goal class
        '''
        self.client = client
        self.goals = []
        self.num_goals = num_goals

        #instantiate the goals
        duck_scale = [1, 1, 1]
        shift = [0, -0.02, 0]
        visualShapeId = self.client.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName="duck.obj",
                                            rgbaColor=[1, 1, 1, 1],
                                            specularColor=[0.4, 0.4, 0],
                                            visualFramePosition=shift,
                                            meshScale=duck_scale)

        collisionShapeId = self.client.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName="duck_vhacd.obj",
                                                collisionFramePosition=shift,
                                                meshScale=duck_scale)

        #Generate random coordinates for the goal
        x, y = [], []
        for i in range(num_goals):
            x.append(random.randrange(-9, 9, 2))
            y.append(random.randrange(1, 19, 2))
            self.client.createMultiBody(baseMass=1,
                            baseInertialFramePosition=[0, 0, 0],
                            baseCollisionShapeIndex=collisionShapeId,
                            baseVisualShapeIndex=visualShapeId,
                            baseOrientation=[0, 45, 45, 0],
                            basePosition=[x[i], y[i], 0])
            self.goals.append([x[i], y[i]])

    def get_goals(self):
        '''
        Function to return the goals
        '''
        return self.goals
