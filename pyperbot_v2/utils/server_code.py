import socket
import os
import numpy as np

def rearrange_array(old_array, joint_values):
    '''
    Methods to rearrange info from the csv file into 4x5 array
    '''
    joint_values[0] = old_array[0:5]
    joint_values[1] = old_array[5:10]
    joint_values[2] = old_array[10:15]
    joint_values[3] = old_array[15:20]
    return joint_values

def setup_server(host, port):
    '''
    #code for establishing socket server
    '''
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket has been created")

    try:
        s.bind((host, port))
    
    except socket.error as msg:
        print(msg)
    
    print("Socket bind complete.")

    return s

def setup_connection(s):
    s.listen(1)
    conn, address = s.accept()
    print("Connected to: " +address[0]+ ": " +str(address[1]))
    return conn

def data_transfer(conn, transmitting_data):
    while True:
        #Receive data from the snake.
        data = conn.recv(1024)
        data = data.decode('utf-8')

        dataMessage = data.split(' ', 1)
        command = dataMessage[0]

        if(command == 'REQ'):
            reply = transmitting_data
        
        elif(command == 'EXIT'):
            print("Client has left.")
            break

        elif(command == 'KILL'):
            print("Our server is shutting down.")
            s.close()
            break
        else:
            reply = "Unknown command."
        
        conn.sendall(str.encode(reply))
    conn.close()