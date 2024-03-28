import pybullet as p
import math

class Gaits():
    def __init__(self, robot_id):
        # Physics setting
        self._anisotropic_friction = [0.005, 0.005, 1]
        self._lateral_friction = 2
        self._robot_id = robot_id
        self._wave_front = 0
        for i in range(-1, p.getNumJoints(self._robot_id)):
            p.changeDynamics(robot_id, i, lateralFriction = self._lateral_friction, anisotropicFriction=self._anisotropic_friction)
        return
    
    #define friction getters and setters
    def get_anisotropic_friction(self):
        return self._anisotropic_friction
    
    def get_lateral_friction(self):
        return self._lateral_friction
    
    def set_anisotropic_friction(self, anisotropicFriction):
        self._anisotropic_friction = anisotropicFriction
    
    def set_lateral_friction(self, lateralFriction):
        self._lateral_friction = lateralFriction

    def _get_keyboard_input(self):
        # Map direction to steering. Up: 65297, Down: 65298, Right: 65296, Left: 65295
        input_dict = {65297: 1j, 65298: -1j, 65296:  1, 65295: -1}
        steering_dict = {1: 0.8, -1: -0.8, 1j: 0, -1j: 'x', 1+1j: 0.8, -1+1j : -0.8}

        pressed_keys, direction = [], 0
        key_codes = p.getKeyboardEvents().keys()
        for key in key_codes:
            if key in input_dict:
                pressed_keys.append(key)
                direction += input_dict[key]

        steering = steering_dict.get(direction, 0)

        if steering == 'x':
            for i in range(p.getNumJoints(self._robot_id)):
                p.setJointMotorControl2(self._robot_id, i, p.POSITION_CONTROL, targetPosition=0, force=30)
            self._wave_front = 0
            steering = 0
        
        return steering
    
    def lateral_undulation(self):
        wave_length = 2.05
        wave_period = 1 
        dt = 1./240. 
        wave_amplitude = 1 
        segment_length = 0.4
        scale_start = 1.0
        if (self._wave_front < segment_length * 4.0):
            scale_start = self._wave_front / (segment_length * 4.0)

        joint_list=[7, 16, 25, 34, 3, 12, 21, 30]
        steer_list=[3, 12, 21, 30]
        steering = self._get_keyboard_input()

        for joint in range(p.getNumJoints(self._robot_id)):
            segment = joint
            phase = (self._wave_front - (segment + 1) * segment_length) / wave_length
            phase -= math.floor(phase)
            phase *= math.pi * 2.0

            # map phase to curvature
            if joint in joint_list:
                target_pos = math.sin(phase)* scale_start* wave_amplitude + steering
            elif joint in steer_list:
                target_pos = steering
            else:
                target_pos = 0

            p.setJointMotorControl2(self._robot_id, joint, p.POSITION_CONTROL, targetPosition=target_pos, force=30)
        self._wave_front += dt/wave_period*wave_length

    def rectilinear_locomotion(self):
        wave_length = 2
        wave_period = 1
        dt = 1./240. 
        wave_amplitude = 0.25
        segment_length = 0.4
        scale_start = 1
        
        if (self._wave_front < segment_length * 4.0):
            scale_start = self._wave_front / (segment_length * 4.0)

        joint_list = [8, 17, 26, 35]
        steer_list = [3, 12, 21, 30]
        steering = self._get_keyboard_input()

        for joint in range(p.getNumJoints(self._robot_id)):
            
            segment = joint
            phase = (self._wave_front - (segment + 1) * segment_length) / wave_length
            phase -= math.floor(phase)
            phase *= math.pi * 2.0
            
            #map phase to curvature
            if joint in joint_list:
                target_pos = math.sin(phase)* scale_start* wave_amplitude
            elif joint in steer_list:
                # Steering currently disabled for concertina locomotion.
                target_pos = steering
                #target_pos = 0
            else:
                target_pos = 0

            p.setJointMotorControl2(self._robot_id, joint, p.POSITION_CONTROL, targetPosition=target_pos, force=30)
        
        self._wave_front += dt/wave_period*wave_length

    
    def rectilinear_locomotion_irl(self):
        wave_length = 2
        wave_period = 1
        dt = 1./240. 
        wave_amplitude = 0.5
        segment_length = 0.4
        scale_start = 1
        
        if (self._wave_front < segment_length * 4.0):
            scale_start = self._wave_front / (segment_length * 4.0)

        joint_list = [8, 17, 26, 35, 2, 11, 20, 29]
        steer_list = [3, 12, 21, 30]
        steering = self._get_keyboard_input()

        for joint in range(p.getNumJoints(self._robot_id)):
            
            segment = joint
            phase = (self._wave_front - (segment + 1) * segment_length) / wave_length
            phase -= math.floor(phase)
            phase *= math.pi * 2.0
            
            #map phase to curvature
            if joint in joint_list:
                target_pos = math.sin(phase)* scale_start* wave_amplitude
            elif joint in steer_list:
                # Steering currently disabled for concertina locomotion.
                target_pos = steering*4
                #target_pos = 0
            else:
                target_pos = 0

            p.setJointMotorControl2(self._robot_id, joint, p.POSITION_CONTROL, targetPosition=target_pos, force=30)
        
        self._wave_front += dt/wave_period*wave_length