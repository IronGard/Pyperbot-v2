#Imports
import socket
import ast
import numpy as np

#Setup
#Gene's house wifi's ip address: 192.168.1.88
#Hahn's laptop connected to Gene's hotspot: 192.168.97.1
#Gene's laptop connected to Gene's hotspot: 192.168.97.28
host = '192.168.1.88' #IPv4 address
port = 6969

#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.connect((host,port))
joint_angles_1d = np.zeros(20)
joint_angles_2d = np.zeros((4,5))

counter = 0


class client():
    def __init__(self, port, host):
        self.port = port
        self.host = host
        self.server = self.setup_server()
        
        #Received message
        self.received_command = ""
        self.received_data = ""
        self.received_message_status = "UNREAD"
        
        #Transmitting message
        self.transmitting_command = ""
        self.transmitting_data = ""
    
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
    
    #Function for FETCHing and DECODing the data received from the server
    def receive_message(self):
        received_message = self.server.recv(1024)
        received_message = received_message.decode('utf-8')
        received_message = received_message.split(" ")
        
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
            self.transmitting_command = "REQ_JOINT_POS"
            self.transmitting_data = "NULL"
        
        elif(self.received_command == "REQ_SENSOR_DATA"):
            print("Sending back sensor data")
            self.transmitting_command = "REQ_JOINT_POS"
            self.transmitting_data = "NULL"
        
        elif(self.received_command == "MOVE_JOINT"):
            print("Moving joints")
            self.transmitting_command = "JOINT_MOVED"
            self.transmitting_data = "NULL"
        
        #Encoding the message to be transmitted back to the server
        transmitting_message = self.transmitting_command + " " + self.transmitting_data + " " + "UNREAD"
        self.server.send(str.encode(transmitting_message))

#Creates my_client object, with the specified port and host
my_client = client(port, host)

#Connection is valid while true
while True:
    my_client.receive_message()
    
    if (my_client.get_message_status() == "UNREAD"):
        my_client.execute_message()

