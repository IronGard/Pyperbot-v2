#standardised snakebot class for hyperparameter tuning process
'''
Primary Changes:
* One goal with standardised position
* Snake position standardised.
'''

#updated file to match new snakerobot design
#TODO: add option to change between different snakebots directly
#TODO: add reward as the distance to the closest goal, not first goal

import pybullet as p
import numpy as np 
import math
import os
import sys
import random
import configparser
from gymnasium.utils import seeding

#import utils functions for loading goals and maze
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from utils.structure_loader import Loader
from utils.snakebot_info import Info


class StandardSnakebot:
    def __init__(self, client, basePosition = [0, 0, 0], baseOrientation = [0, 0, 0, 1], 
                 snakebot_dir = "pyperbot_v2/snakebot_description/urdf/updated_full_snakebot_no_macro.urdf.xacro",
                 gait = "lateral_undulation", manual = False, seed = 0):
        '''
        Constructor for the Snakebot class to define a snakebot object. Altered to allow for additional parameters.

        Args:
            client: the pybullet client
            basePosition: the position of the base of the snakebot
            baseOrientation: the orientation of the base of the snakebot
            snakebot_dir: the directory of the snakebot urdf file
            gait: the gait of the snakebot
            manual: whether the snakebot is being controlled manually
        '''
        self.seed_value = 0
        self.set_seed(seed)
        self._client = client
        self._gait = gait
        self._manual = manual
        self._robot = self.genSnakebot(snakebot_dir, basePosition, baseOrientation)
        self._info = Info(self.get_ids()[0])

    def get_seed(self):
        return self.seed_value
    
    def set_seed(self, seed = 0):
        self.seed_value = seed

    def get_ids(self):
        '''
        Returns id of the robot and the plane used
        '''
        return self._robot, self._client._client
    
    def apply_action(self, actions):
        '''
        Function to apply actions to the snakebot
        To be further refined and implemented based on actual snake robot movement.
        * Should be more precise for handling revolute and prismatic joints in particular.
        * Need to decide how to map joint movements to the snakebot's movement.
        '''
        #action needs to be twenty dimensional
        #consider setting prismatic and revolute joints separately.
        #check why the render doesn't work with the client for some reason - make sure that the joint values are still being printed out appropriately at the end of the simulation.
        #Each time step is 1.240 of a second
        #we can calcualte the drag and friction of the car as well.
        #debug - using p.setjointMotorControl2 to set joint motor control
        moving_joint_list = [joint for joint in range(p.getNumJoints(self._robot)) if p.getJointInfo(self._robot, joint)[2] != p.JOINT_FIXED]
        for i in range(len(moving_joint_list)):
            self._client.setJointMotorControl2(self._robot, 
                                    moving_joint_list[i], 
                                    p.POSITION_CONTROL, 
                                    targetPosition = actions[i], 
                                    force = 30)
    
    
    def gen_goals(self, num_goals = 1):
        '''
        Function to generate number of goals to be added to the snakebot environment
        '''
        goal = [5, 0, 0]
        duck_scale = [1, 1, 1]
        shift = [0, -0.02, 0]
        visualShapeId = self._client.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName="duck.obj",
                                            rgbaColor=[1, 1, 1, 1],
                                            specularColor=[0.4, 0.4, 0],
                                            visualFramePosition=shift,
                                            meshScale=duck_scale)

        collisionShapeId = self._client.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName="duck_vhacd.obj",
                                                collisionFramePosition=shift,
                                                meshScale=duck_scale)

        #Generate random coordinates for the goal
        
        self._client.createMultiBody(baseMass=1,
                            baseInertialFramePosition=[0, 0, 0],
                            baseCollisionShapeIndex=collisionShapeId,
                            baseVisualShapeIndex=visualShapeId,
                            baseOrientation=[0, np.pi/4, np.pi/4, 0],
                            basePosition=goal)
        return goal
    
    def genSnakebot(self, snakebot_dir, basePosition, baseOrientation):
        '''
        Function to generate the snakebot in the environment
        '''
        robot = self._client.loadURDF(fileName = snakebot_dir, 
                                      basePosition = [0, 0, 0], 
                                      baseOrientation = [0, 0, 0, -np.pi/2])
        return robot
    
    def get_joint_observation(self):
        #get num moving joints and moving joint indices
        moving_joints_inds = [joint for joint in range(self._client.getNumJoints(self._robot)) if self._client.getJointInfo(self._robot, joint)[2] != p.JOINT_FIXED]
        num_moving_joints = len(moving_joints_inds)
        joint_states = self._client.getJointStates(self._robot, moving_joints_inds)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        return [joint_positions, joint_velocities]
    
    def get_base_observation(self):
        '''
        Obtain the position and orientation of the robot and use these parameters as an observation of the state of the environment.
        Compute the sum to return an individual float as the observation value.
        '''
        pos, ang = self._client.getBasePositionAndOrientation(self._robot)
        ori = self._client.getEulerFromQuaternion(ang)
        return [pos, ori]
    
    def get_observation(self):
        '''
        return full observation in desired format        
        '''
        pos, ang = self._client.getBasePositionAndOrientation(self._robot)
        ori = p.getEulerFromQuaternion(ang)
        base_velocity, _ = self._client.getBaseVelocity(self._robot)
        linear_velocity = np.linalg.norm(base_velocity) #linear velocity
        observation = np.array(list(pos) + list(ori) + [linear_velocity], dtype = np.float32)
        return observation
    
    
