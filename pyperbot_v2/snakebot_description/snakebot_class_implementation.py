#File to define the snakebot with a class-based implementation
import pybullet as p
import numpy as np 
import math
import os

class Snakebot:
    def __init__(self, client):
        '''
        Constructor for the Snakebot class to define a snakebot object
        '''
        self.client = client
        robot_name = os.path.join(os.getcwd(), 'pyperbot_v2/snakebot_description/urdf/updated_snakebot.urdf.xacro')
        print(robot_name)
        self.robot = p.loadURDF(robot_name, [0, 0, 0], globalScaling = 2.0, physicsClientId = client)
        self.planeID = p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId = client)
        #getting prismatic joints and revolute joints from the robot
        self.moving_joints = [p.getJointInfo(self.robot, joint, physicsClientId = client) for joint in range(p.getNumJoints(self.robot, physicsClientId = client)) if p.getJointInfo(self.robot, joint, physicsClientId = client)[2] != p.JOINT_FIXED]
        self.moving_joints_inds = [p.getJointInfo(self.robot, joint, physicsClientId = client)[0] for joint in range(p.getNumJoints(self.robot, physicsClientId = client)) if p.getJointInfo(self.robot, joint, physicsClientId = client)[2] != p.JOINT_FIXED]
        print(self.moving_joints_inds)
        self.prismatic_joints = [joint for joint in range(p.getNumJoints(self.robot, physicsClientId = client)) if p.getJointInfo(self.robot, joint, physicsClientId = client)[2] == p.JOINT_PRISMATIC]
        self.revolute_joints = [joint for joint in range(p.getNumJoints(self.robot, physicsClientId = client)) if p.getJointInfo(self.robot, joint, physicsClientId = client)[2] == p.JOINT_REVOLUTE]
        self.num_joints = len(self.moving_joints)
    def get_ids(self):
        '''
        Returns id of the robot and the plane used
        '''
        return self.robot, self.planeID
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
        moving_joint_list = [0, 1, 2, 5, 6, 8, 9, 10, 13, 14, 16, 17, 18, 21, 22, 24, 25, 26, 29, 30]
        for i in range(len(moving_joint_list)):
            p.setJointMotorControl2(self.robot, 
                                    moving_joint_list[i], 
                                    p.POSITION_CONTROL, 
                                    targetPosition = actions[i], 
                                    force = 30, 
                                    physicsClientId = self.client)
        # p.setJointMotorControlArray(self.robot,
        #                             [0, 1, 2, 5, 6, 8, 9, 10, 13, 14, 16, 17, 18, 21, 22, 24, 25, 26, 29, 30],
        #                             p.POSITION_CONTROL,
        #                             targetPositions = actions,
        #                             forces = [30]*self.num_joints,
        #                             physicsClientId = self.client)
        # print(len([0, 1, 2, 5, 6, 8, 9, 10, 13, 14, 16, 17, 18, 21, 22, 24, 25, 26, 29, 30]))
        # print(len(actions))
    
    def get_joint_observation(self):
        #get num moving joints and moving joint indices
        moving_joints_inds = [joint for joint in range(p.getNumJoints(self.robot, physicsClientId = self.client)) if p.getJointInfo(self.robot, joint, physicsClientId = self.client)[2] != p.JOINT_FIXED]
        num_moving_joints = len(moving_joints_inds)
        joint_states = p.getJointStates(self.robot, moving_joints_inds, physicsClientId = self.client)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        return [joint_positions, joint_velocities]
    
    def get_base_observation(self):
        '''
        Obtain the position and orientation of the robot and use these parameters as an observation of the state of the environment.
        Compute the sum to return an individual float as the observation value.
        '''
        pos, ang = p.getBasePositionAndOrientation(self.robot, self.client)
        ori = p.getEulerFromQuaternion(ang)
        return [pos, ori]
    
