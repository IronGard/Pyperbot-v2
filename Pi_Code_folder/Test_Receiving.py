import socket
import ast
import numpy as np

host = "192.168.97.28"
port = 6969

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host,port))
print(f"Connection received at {host}")

joint_angles_1d = np.zeros(20)
joint_angles_2d = np.zeros((4,5))

counter = 0
reply = ""

while True:
    received_command = s.recv(1024)
    
    if(received_command == "DEBUG"):
        print("Move to initial position.")
    
    elif(received_command == "REQ_JOINT"):
        







while True:
    
    received_command = s.recv(1024)
    
    if (received_command == "IDLE"):
        command = "REQ_JOINT_POS"
        s.send(str.encode(command))
        
    
    #command = "REQ_JOINT_POS"
    #s.send()

#while (reply != "KILL"):
    #command = "REQ"
    #s.send(str.encode(command))
    #reply = s.recv(1024)
    #reply = (reply.decode('utf-8'))
    #split_reply = reply.split(',')
    #print(len(split_reply))

    #for i in range(len(split_reply)):
        #joint_angles_1d[i] = float(split_reply[i])
        
    #joint_angles_2d = np.reshape(np.array(joint_angles_1d), (4, 5))
    #print(joint_angles_2d)
    #input()
    #counter = counter + 1
s.close()

print(counter)

