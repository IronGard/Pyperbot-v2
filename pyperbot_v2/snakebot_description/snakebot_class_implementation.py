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
        self.mazeID = p.loadURDF("maze.urdf")
        #getting prismatic joints and revolute joints from the robot
        self.moving_joints = [p.getJointInfo(self.robot, joint, physicsClientId = client) for joint in range(p.getNumJoints(self.robot, physicsClientId = client)) if p.getJointInfo(self.robot, joint, physicsClientId = client)[2] != p.JOINT_FIXED]
        self.prismatic_joints = [joint for joint in range(p.getNumJoints(self.robot, physicsClientId = client)) if p.getJointInfo(self.robot, joint, physicsClientId = client)[2] == p.JOINT_PRISMATIC]
        self.revolute_joints = [joint for joint in range(p.getNumJoints(self.robot, physicsClientId = client)) if p.getJointInfo(self.robot, joint, physicsClientId = client)[2] == p.JOINT_REVOLUTE]
        self.num_joints = len(self.prismatic_joints) + len(self.revolute_joints)
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
        p.setJointMotorControlArray(self.robot,
                                    range(20),
                                    p.POSITION_CONTROL,
                                    targetPositions = [actions]*20,
                                    forces = [30]*self.num_joints,
                                    physicsClientId = self.client)
    
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
    
    def get_position(self):
        '''
        returns base position of the robot specifically. Helper function for get_dist_to_goal.
        '''
        return p.getBasePositionAndOrientation(self.robot, self.client)[0]
    
    def get_model_observation(self):
        '''
        Function to return reward and observation based on required format in TestEnv.
        Returns 10 dimensional array:
        * Base position (x, y, z)
        * Base orientation (r, p, y)
        * Remaining distance to goal
        * Linear velocity of the robot.
        '''
        joint_obs = self.get_joint_observation()