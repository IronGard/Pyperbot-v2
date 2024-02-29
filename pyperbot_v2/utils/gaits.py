import pybullet as p
import math

class Gaits():
    def __init__(self, robot_id):
        # Physics setting
        anisotropicFriction = [0.005, 0.005, 1]
        lateralFriction = 2
        self._robot_id = robot_id
        self._wave_front = 0
        for i in range(-1, p.getNumJoints(self._robot_id)):
            p.changeDynamics(robot_id, i, lateralFriction = lateralFriction,anisotropicFriction=anisotropicFriction)
        return
    
    def _get_keyboard_input(self):
        # Map direction to steering. Up: 65297, Down: 65298, Right: 65296, Left: 65295
        input_dict = {65297: 1j, 65298: -1j, 65296:  1, 65295: -1}
        steering_dict = {1: 0.8, -1: -0.8, 1j: 0, -1j: 0, 1+1j: 0.4, -1+1j : -0.4}

        pressed_keys, direction = [], 0
        key_codes = p.getKeyboardEvents().keys()
        for key in key_codes:
            if key in input_dict:
                pressed_keys.append(key)
                direction += input_dict[key]

        steering = steering_dict.get(direction, 0)

        return steering
    
    def lateral_undulation(self):
        waveLength = 4.2 
        wavePeriod = 1 
        dt = 1./240. 
        waveAmplitude = 1 
        segmentLength = 0.4 
        scaleStart = 1.0
        if (self._wave_front < segmentLength * 4.0):
            scaleStart = self._wave_front / (segmentLength * 4.0)

        joint_list = [2, 6, 10, 14, 18, 22, 26, 30]
        steering = self._get_keyboard_input()

        for joint in range(p.getNumJoints(self._robot_id)):
            segment = joint
            phase = (self._wave_front - (segment + 1) * segmentLength) / waveLength
            phase -= math.floor(phase)
            phase *= math.pi * 2.0

            #map phase to curvature
            targetPos = math.sin(phase)* scaleStart* waveAmplitude + steering
            if joint not in joint_list:
                targetPos = 0
            p.setJointMotorControl2(self._robot_id, joint, p.POSITION_CONTROL, targetPosition=targetPos, force=30)

        self._wave_front += dt/wavePeriod*waveLength