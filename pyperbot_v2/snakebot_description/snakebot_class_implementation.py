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
        self.prismatic_joints = [joint for joint in range(p.getNumJoints(self.robot, physicsClientId = client)) if p.getJointInfo(self.robot, joint, physicsClientId = client)[2] == p.JOINT_PRISMATIC]
        self.revolute_joints = [joint for joint in range(p.getNumJoints(self.robot, physicsClientId = client)) if p.getJointInfo(self.robot, joint, physicsClientId = client)[2] == p.JOINT_REVOLUTE]

    def get_ids(self):
        '''
        Returns id of the robot and the plane used
        '''
        return self.robot, self.planeID
    def apply_action(self, action):
        '''
        Function to apply actions to the snakebot
        To be further refined and implemented based on actual snake robot movement.
        * Should be more precise for handling revolute and prismatic joints in particular.
        '''
        for joint in range(len(action)):
            p.setJointMotorControl2(self.robot,
                                    joint,
                                    p.POSITION_CONTROL,
                                    targetPosition=action[joint],
                                    force=30,
                                    physicsClientId = self.client)
    def get_joint_observation(self):
        #get num moving joints and moving joint indices
        moving_joints_inds = [joint for joint in range(p.getNumJoints(self.robot, physicsClientId = self.client)) if p.getJointInfo(self.robot, joint, physicsClientId = self.client)[2] != p.JOINT_FIXED]
        num_moving_joints = len(moving_joints_inds)
        joint_states = p.getJointStates(self.robot, moving_joints_inds, physicsClientId = self.client)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques
    
    def get_base_observation(self):
        pos, ang = p.getBasePositionAndOrientation(self.robot, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        pos = pos[:2]

        vel = p.getBaseVelocity(self.robot, self.client)
        observation = (pos + ori + vel)
        return observation
    
    def get_position(self):
        '''
        returns base position of the robot
        '''
        return p.getBasePositionAndOrientation(self.robot, self.client)[0]