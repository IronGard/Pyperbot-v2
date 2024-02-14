#File to define the snakebot with a class-based implementation
import pybullet as p
import numpy as np 
import os

class Snakebot:
    def __init__(self, client):
        self.client = client
        robot_name = os.path.join(__file__, 'snakebot_description/urdf/updated_snakebot.urdf.xacro')
        self.robot = p.loadURDF(robot_name, [0, 0, 0], globalScaling = 2.0, physicsClientId = client)
        self.planeID = p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId = client)
        #getting prismatic joints and

    def get_ids(self):
        return self.robot, self.planeID
    def apply_action(self, action):
        for joint in range(len(action)):
            p.setJointMotorControl2(self.robot,
                                    joint,
                                    p.POSITION_CONTROL,
                                    targetPosition=action[joint],
                                    force=30,
                                    physicsClientId = self.client)
    def get_joint_observation(self):
        joint_states = p.getJointStates(self.robot, range(p.getNumJoints(self.robot)), physicsClientId = self.client)
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