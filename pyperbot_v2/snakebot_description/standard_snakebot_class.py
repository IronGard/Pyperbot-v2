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
    def __init__(self, client, basePosition = [0, 0, 0.5], baseOrientation = [0, 0, 0, 1], 
                 snakebot_dir = "pyperbot_v2/snakebot_description/urdf/snakebot_swapped.urdf.xacro",
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
    
    #unnormalise action function
    def unnormalise_action(self, action):
        '''
        Function to unnormalise the action space from the range [-1, 1] to the actual joint limits
        '''
        #get joint limits
        lateral_undulation_joint_list = [4, 7, 13, 16, 22, 25, 34, 3, 12, 21, 30]
        moving_joint_list = [2, 3, 8, 11, 12, 17, 20, 21, 26, 29, 30, 35]
        all_moving_joint_ids = [2, 3, 4, 7, 8, 11, 12, 13, 16, 17, 20, 21, 22, 25, 26, 29, 30, 31, 34, 35]
        #get joint limits only for joints in moving joint list
        joint_limits = [self._client.getJointInfo(self._robot, joint)[8:10] for joint in all_moving_joint_ids]
        # print(joint_limits)
        #unnormalise actions
        unnormalised_actions = [action[i]*(joint_limits[i][1]-joint_limits[i][0])/2 + (joint_limits[i][1]+joint_limits[i][0])/2 for i in range(len(action))]
        return unnormalised_actions
    
    def apply_action(self, actions):
        '''
        Function to apply actions to the snakebot
        To be further refined and implemented based on actual snake robot movement.
        * Should be more precise for handling revolute and prismatic joints in particular.
        * Need to decide how to map joint movements to the snakebot's movement.
        '''
        #fixed actions to be focusing on the eight primary joints responsible for the propagation of the snake.
        
        #renormalise actions from joint limits
        actions = self.unnormalise_action(actions)
        #scale down when using velocity control
        # actions = [action*10 for action in actions]
        lateral_undulation_joint_list = [7, 16, 25, 34, 3, 12, 21, 30]
        moving_joint_list = [2, 3, 8, 11, 12, 17, 20, 21, 26, 29, 30, 35]
        all_moving_joint_ids = [2, 3, 4, 7, 8, 11, 12, 13, 16, 17, 20, 21, 22, 25, 26, 29, 30, 31, 34, 35]
        # print("actions = ", actions)
        # print("%======================================%\n")
        counter = 0
        joint_list = []
        for joint in range(self._client.getNumJoints(self._robot)):
            if joint in all_moving_joint_ids:
                self._client.setJointMotorControl2(self._robot, 
                                        all_moving_joint_ids[counter], 
                                        p.POSITION_CONTROL, 
                                        targetPosition = actions[counter], 
                                        force = 30)
                counter += 1
        #alternative: using velocity control instead of position control
        # for joint in range(self._client.getNumJoints(self._robot)):
        #     if joint in all_moving_joint_ids:
        #         self._client.setJointMotorControl2(self._robot, 
        #                                 all_moving_joint_ids[counter], 
        #                                 p.VELOCITY_CONTROL, 
        #                                 targetVelocity = actions[counter], 
        #                                 force = 0.1)
        #         counter += 1
        #using joint motor control array
        # self._client.setJointMotorControlArray(self._robot, lateral_undulation_joint_list, controlMode = p.VELOCITY_CONTROL, targetVelocities = actions, forces = 30*np.ones(len(actions)))
        #final alternative: torque control
        # for joint in range(self._client.getNumJoints(self._robot)):
        #     if joint in all_moving_joint_ids:
        #         joint_list.append(joint)
        #         self._client.setJointMotorControl2(self._robot, 
        #                         all_moving_joint_ids[counter], 
        #                         p.TORQUE_CONTROL, 
        #                         force = actions[counter])
        #         counter += 1
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
                                      basePosition = basePosition, 
                                      baseOrientation = [0, 0, 0, 1])
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
    

    def get_full_observation(self):
        '''
        return full observation in desired format        
        '''
        # pos, ang = self._client.getBasePositionAndOrientation(self._robot)
        # ori = p.getEulerFromQuaternion(ang)
        # base_velocity, _ = self._client.getBaseVelocity(self._robot)
        # linear_velocity = np.linalg.norm(base_velocity)/100 #scaled lin velocity
        joint_positions = self.get_joint_observation()[0]
        # observation = np.array(list(pos) + list(ori) + joint_positions, dtype = np.float32)
        observation = np.array(joint_positions, dtype = np.float32)
        return observation
    
