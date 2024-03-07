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
import ast

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
#Gene's house wifi's ip address: 192.168.1.88
#Hahn's laptop connected to Gene's hotspot: 192.168.97.1
#Gene's laptop connected to  Gene's hotspot: 192.168.97.28
host = '192.168.97.28' #IPv4 address
port = 6969 

#========================|Classes for Snake|========================
class module():
    def __init__(self, address, offsets):
        self.address = address
        self.kit = ServoKit(channels = 16, address = address)
        self.servos = [self.kit.servo[0], self.kit.servo[1], self.kit.servo[2], self.kit.servo[3], self.kit.servo[4]]
        self.servo_angles = np.array([180, 180, 90, 90, 90])
        self.joint_values = np.zeros(5)
        self.offsets = offsets
    
    def set_servo_angle(self, servo_number, new_angle):
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
            self.set_servo_angle(int(joint_number), np.degrees(new_value) + 90)
            self.set_joint_value(int(joint_number), new_value + 90)
            

class snake():
    def __init__(self):
        self.modules = []
        self.number_of_modules = 0
    
    def add_module(self, new_module):
        self.modules.append(new_module)
        self.number_of_modules = self.number_of_modules + 1
    
    def move_module(self, module_number, new_joint_values):
        old_joint_values = self.modules[module_number].get_joint_values()
        moved_joint_number = 0 #index of the joint that is being moved
        
        for i in range(5):
            if(old_joint_values[i] != new_joint_values[i]):
                self.modules[module_number].move_joint(int(i), new_joint_values[i])
    
    def move_to_init_position(self):
        for module_number in range(self.number_of_modules):
            for joint_number in range(5):
                self.modules[module_number].move_joint(joint_number, 0)

#========================|Classes for Client|========================
class client():
    def __init__(self, port, host, snake):
        self.port = port
        self.host = host
        self.snake = snake
        self.server = self.setup_server()
        
        #Received message
        self.received_command = ""
        self.received_data = ""
        self.received_message_status = "UNREAD"
        
        #Transmitting message
        self.transmitting_command = ""
        self.transmitting_data = ""
        
        #Joint values
        self.joint_values_2d = np.zeros((4,5))
        
    #Function for setting up the socket on the client side
    def setup_server(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host,self.port))
        return s
    
    #Function for updating the current command that is being executed
    def set_command(self, new_command):
        self.transmitting_command = new_command
    
    #Function for updating the data that will be transmitted back to the server
    def set_data(self, new_data):
        self.transmitting_data = new_data
    
    def split_string(self, unsplit_string):
        print(unsplit_string)
        split_string = unsplit_string.split(',')
        
        for i in range(len(split_string)):
            split_string[i] = float(split_string[i])
        
        self.joint_values_2d = np.reshape(np.array(split_string), (4,5))
    
    #Function for FETCHing and DECODing the data received from the server
    def receive_message(self):
        received_message = self.server.recv(1024)
        received_message = received_message.decode('utf-8')
        received_message = received_message.split(" ")
        
        print(received_message)
        
        #Message is split into COMMAND, DATA, and MESSAGE_STATUS, and updated to the server object
        self.received_command = received_message[0]
        self.received_data = received_message[1]
        self.received_message_status = received_message[2]
    
    #Function for getting the message status. If the message status = UNREAD, the server will decode and execute it.
    def get_message_status(self):
        return self.received_message_status
    
    #Function for executing the received command
    def execute_message(self):
        #Updates the message status first, so it doesn't get re-executed twice
        self.received_message_status = "READ"
        
        #Command if's are listed in their priorities. DEBUG has the highest priority, so it comes first. Then REQ_SENSOR_DATA, etc.
        if(self.received_command == "DEBUG"):
            print("Moving joints to initial state.")
            self.snake.move_to_init_position()
            self.transmitting_command = "REQ_JOINT_POS"
            self.transmitting_data = "NULL"
        
        elif(self.received_command == "REQ_SENSOR_DATA"):
            print("Sending back sensor data")
            self.transmitting_command = "REQ_JOINT_POS"
            self.transmitting_data = "NULL"
        
        elif(self.received_command == "MOVE_JOINT"):
            print("Moving joints")
            self.split_string(self.received_data)
            print(self.joint_values_2d)
            
            for module_number in range(4):
                self.snake.move_module(module_number, self.joint_values_2d[module_number])
            
            self.transmitting_command = "JOINT_MOVED"
            self.transmitting_data = "NULL"
        
        #Encoding the message to be transmitted back to the server
        transmitting_message = self.transmitting_command + " " + self.transmitting_data + " " + "UNREAD"
        self.server.send(str.encode(transmitting_message))

#========================|Creating Snake Object|========================
my_snake = snake()
my_snake.add_module(module(0x40, MODULE0_OFFSETS))
my_snake.add_module(module(0x41, MODULE0_OFFSETS))
my_snake.add_module(module(0x42, MODULE0_OFFSETS))
my_snake.add_module(module(0x43, MODULE0_OFFSETS))

#========================|Creating Client Object|========================
#Creates my_client object, with the specified port and host
my_client = client(port, host, my_snake)

#Connection is valid while true
while True:
    my_client.receive_message()
    
    if (my_client.get_message_status() == "UNREAD"):
        my_client.execute_message()

