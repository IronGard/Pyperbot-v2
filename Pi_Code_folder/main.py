#========================|General Notations|========================

#All revolute joints have range (-pi/2, pi/2), and take radian inputs

#All primatic joints have range ()

#Joint 0: Spring steel prismatic

#Joint 1: Spring-steel rotation (axis of rotation is perpendicular to the GND plane)

#Joint 2: Universal joint rotation

#Joint 3: C-bracket rotation (axis of rotation is perpendicular to the GND plane)

#Joint 4: C-bracket rotation (axis of rotation is parallel to the GND plane)


#========================|Imports|========================
import time
import board
from adafruit_servokit import ServoKit
import math
import numpy as np
import socket

#========================|Constants|========================

#Mathematical Constants
PI = math.pi

#Measurements(mm)
MODULE_DIAMETER = 100
SPINDLE_INNER_RADIUS = 8
SPINDLE_INNER_PERIMETER = 2*PI*SPINDLE_INNER_RADIUS

#Servo Offsets (deg) 
MODULE0_OFFSETS = [0, 0, 0, 0, 15]

#Server Settings
host = "192.168.97.28"
port = 6969

#========================|Classes|========================

class module():
    def __init__(self, address, offsets):
        self.address = address
        self.kit = ServoKit(channels = 16, address = address)
        self.servos = [self.kit.servo[0], self.kit.servo[1], self.kit.servo[2], self.kit.servo[3], self.kit.servo[4]]
        self.servo_angles = np.array([180, 180, 90, 90, 90])
        self.joint_values = np.zeros(5)
        self.offsets = offsets
    
    def set_servo_angle(self, servo_number, new_angle):
        print(new_angle + self.offsets[servo_number])
        self.servos[servo_number].angle = new_angle + self.offsets[servo_number]
        self.servo_angles[servo_number] = float(new_angle)
    
    def get_servo_angle(self, servo_number):
        return self.servo_angles[servo_number]
    
    def set_joint_value(self, joint_number, new_value):
        self.joint_values[int(joint_number)] = float(new_value)
    
    def get_joint_values(self):
        return self.joint_values
    
    def move_joint(self, joint_number, new_value):
        
        if(joint_number == 0):
            servo_angle = 180 - new_value*360/SPINDLE_INNER_PERIMETER
            self.set_servo_angle(0, servo_angle)
            self.set_servo_angle(1, servo_angle)
            self.set_joint_value(0, new_value)
        
        elif(joint_number == 1):
            delta_d = MODULE_DIAMETER * math.tan(new_value)
            servo_angle = delta_d * 360 / SPINDLE_INNER_PERIMETER
        
            if(new_value < 0):
                self.set_servo_angle(0, 180)
                self.set_servo_angle(1, 180 + servo_angle)
        
            else:
                self.set_servo_angle(0, 180 - servo_angle)
                self.set_servo_angle(1, 180)
        
            self.set_joint_value(1, new_value)
        
        else:
            print((new_value))
            self.set_servo_angle(int(joint_number), np.degrees(new_value) + 90)
            self.set_joint_value(int(joint_number), new_value + 90)
            

class snake():
    def __init__(self):
        self.modules = []
    
    def add_module(self, new_module):
        self.modules.append(new_module)
    
    def move_module(self, module_number, new_joint_values):
        old_joint_values = self.modules[module_number].get_joint_values()
        moved_joint_number = 0 #index of the joint that is being moved
        
        for i in range(5):
            if(old_joint_values[i] != new_joint_values[i]):
                self.modules[module_number].move_joint(int(i), new_joint_values[i])
            

#========================|Creating the Snake Object|========================
my_snake = snake()
my_snake.add_module(module(0x41, MODULE0_OFFSETS))
my_snake.add_module(module(0x41, MODULE0_OFFSETS))
my_snake.add_module(module(0x42, MODULE0_OFFSETS))
my_snake.add_module(module(0x43, MODULE0_OFFSETS))

#========================|Client Code|========================
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host,port))
print(f"Connection received at {host}")

joint_angles_1d = np.zeros(20)
joint_angles_2d = np.zeros((4,5))


counter = 0
reply = None

while (reply != "KILL"):
    command = "REQ"
    s.send(str.encode(command))
    reply = s.recv(1024)
    reply = (reply.decode('utf-8'))
    split_reply = reply.split(',')
    print(len(split_reply))

    for i in range(len(split_reply)):
        joint_angles_1d[i] = float(split_reply[i])
        
    joint_angles_2d = np.reshape(np.array(joint_angles_1d), (4, 5))
    
    for i in range(4):
        my_snake.move_module(i, joint_angles_2d[i])

    input()
    counter = counter + 1
s.close()


#========================|Test Code|========================

module0 = module(0x40, MODULE0_OFFSETS)


#Function for testing sinusoidal motion in joint 5 (bottom c-bracket)
def test_joint_5():
    angle_step = 5
    initial_angle = -30
    current_angle = initial_angle

    while(1):
        while(current_angle < -1*initial_angle):#
            current_angle = current_angle + angle_step
            my_snake.move_module(0, [0, 0, 0, 0, np.radians(current_angle)])
            my_snake.move_module(1, [0, 0, 0, 0, np.radians(current_angle)])
            my_snake.move_module(2, [0, 0, 0, 0, np.radians(current_angle)])
            my_snake.move_module(3, [0, 0, 0, 0, np.radians(current_angle)])

            time.sleep(0.1)
        
        initial_angle = initial_angle*-1
    
        while(current_angle > initial_angle*-1):
            current_angle = current_angle - angle_step
            my_snake.move_module(0, [0, 0, 0, 0, np.radians(current_angle)])
            my_snake.move_module(1, [0, 0, 0, 0, np.radians(current_angle)])
            my_snake.move_module(2, [0, 0, 0, 0, np.radians(current_angle)])
            my_snake.move_module(3, [0, 0, 0, 0, np.radians(current_angle)])
            time.sleep(0.1)
    
        initial_angle = initial_angle*-1    
    
test_joint_5()
        
