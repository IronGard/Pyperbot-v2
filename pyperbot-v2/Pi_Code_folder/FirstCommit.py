#Joint index = 0,1,2,3,4
#Servo index = 0,1,2,3,4

#========================|Imports|========================
import time
import board
from adafruit_servokit import ServoKit
import math
import numpy as np

#========================|Constants|========================

#Mathematical Constants
PI = math.pi

#Measurements(mm)
MODULE_DIAMETER = 100
SPINDLE_INNER_RADIUS = 8
SPINDLE_INNER_PERIMETER = 2*PI*SPINDLE_INNER_RADIUS

#Servo Offsets (deg)
SERVO_OFFSET = [0, 0, 0, 0, 0]

#Setting limit on servo


#========================|User-Defined Classes|========================

#Class for storing information about each module
class Module:
    
    #Constructor
    def __init__(self, module_address):
        self.module_address = module_address
        self.kit = ServoKit(channels = 16, address = module_address)
        self.joint_angles = np.zeros(5)
        self.servo_angles = np.zeros(5)
    
    #Function for setting servo angle, called by set_joint_angle
    def set_servo_angles(self):
        
        #1. Get joint_angles by calling the get_servo_angles function
        temp_joint_angles = self.joint_angles
        temp_joint_angles = np.array(temp_joint_angles)

        #2. Convert temp_joint_angles to degrees (except J0, which is prismatic)
        joint_angles_deg = np.zeros(5)
        joint_angles_deg[0] = temp_joint_angles[0]
        
        for i in range(1,5):
            joint_angles_deg[i] = np.degrees(temp_joint_angles[i])    
        
        #3. Create vector for storing new servo angles, new_servo_angles
        new_servo_angles = np.zeros(5)
        
        #4. Do the math for each element in the joint_angles_deg vector
        
        #Joint 0: Spring steel (P)
        translation = joint_angles_deg[0]
        servo_angle = translation * 360 / SPINDLE_INNER_PERIMETER
        
        new_servo_angles[0] = servo_angle
        new_servo_angles[1] = servo_angle
        
        #Joint 1: Spring steel (R)
        rotation = temp_joint_angles[1] #This is in radians, because math.tan takes radian input
        delta_d = MODULE_DIAMETER * math.tan(rotation) #delta_d is the declination on either side of the module which generates the angle J1
        servo_angle = delta_d * 360 / SPINDLE_INNER_PERIMETER
        
        #Rotate servo2 if we have a positive angle, and servo3 if we have a negative angle
        if(rotation >= 0):
            new_servo_angles[0] = servo_angle
        
        else:
            new_servo_angles[1] = servo_angle
        
        #Joint 2: Universal Joint (R)]
        
        new_servo_angles[2] = joint_angles_deg[2]
        
        #Joint 3: Top C-Bracket (R)
        new_servo_angles[3] = joint_angles_deg[3]
        
        #Joint 4: Bottom C-Bracket (R)
        new_servo_angles[4] = joint_angles_deg[4]
        
        #5. Move servos to the new angles
        for i in range(5):
            self.kit.servo[i].angle = new_servo_angles[i]
        
        #6. Update servo angles by using self.servo_angles = new_servo_angles
        self.servo_angles = new_servo_angles
        
    
    #Function for returning the module's current servo angles
    def get_servo_angles(self):
        temp_servo_angles = self.servo_angles
        print(temp_servo_angles[0])
        return self.servo_angles
    
    #Function for setting joint angle
    def set_joint_angles(self, joint_angles):
        self.joint_angles = joint_angles
        self.set_servo_angles()

    
module0 = Module(0x40)
module0.set_joint_angles([0, 0, 0, 0, 0, 0])
a = module0.get_servo_angles()
time.sleep(1)
module0.set_joint_angles([1, 0, 0, 0, 0, 0])
time.sleep(1)        
            
        
    