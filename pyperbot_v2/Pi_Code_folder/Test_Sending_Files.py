import socket
import numpy as np
import csv

host = '0.0.0.0'
port = 6969
csv_file_path = "dataset/joint_positions.csv"
read_line = ""
send_line = ""
transmitting_data = []

#Setting up the server
def setup_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket has been created.")
    try:
        s.bind((host, port))
    except socket.error as msg:
        print(msg)

    print("Socket bind complete")

    return s

def setup_connection():
    #Listen to only one device
    s.listen(1)
    conn, address = s.accept()
    print("Connected to: " +address[0] + ": " +str(address[1]))
    return conn

#Function for transfering data_to_be_sent to the robot via the connection conn
def data_transfer(conn, transmitting_data):
    n = 0
    while True:
        #Receive data from the snake.data format: 'COMMAND DATA', where COMMAND can be 'REQ' to request next joint angles, 'KILL' to terminate connection, etc.
        data = conn.recv(1024) #1024 = buffer size
        data = data.decode('utf-8')

        dataMessage = data.split(' ', 1)
        command = dataMessage[0]
        #print(len(transmitting_data))

        #Split the data such that you separate the command from the rest of the data
        #command = "REQ"

        #if (n == len(transmitting_data)):
        #    conn.sendall(str.encode("KILL"))
        #If the robot is requesting joint angles, send back data_to_be_sent
        if (command == "REQ"):
            conn.sendall(str.encode(transmitting_data[n]))
            print(transmitting_data[n])
            n = n + 1

        #If command is EXIT, the connection terminates, but the server remains running for other clients to connect to
        elif command == 'EXIT':
            print("Client has left.")
            break

        #If command is KILL, the server ends itself
        elif command == 'KILL':
            print("Our server is shutting down.")
            s.close()

            #Unknown command
        else:
            reply = "Unknown command."

        #Sends encoded reply back to all connected devices (in this case, we only have one device: the snake's RPi)

    conn.close()

#Reading csv file
with open(csv_file_path, "r") as file:
    csv_reader = csv.reader(file)
    counter = 0
    sampling_delay = 10
    starting_point = 50

    for row in csv_reader:
        if(counter >= starting_point):
            if(counter%sampling_delay == 0):
                read_line = row
                send_line = ""
                for i in range(1,20):
                    rounded_read_line = str(round(float(read_line[i]), 2))
                    send_line = send_line + rounded_read_line + ","
                rounded_read_line = str(round(float(read_line[20]), 2))
                send_line = send_line + rounded_read_line
                transmitting_data.append(send_line)
        counter = counter + 1

#Setting up server
s = setup_server()

#Running the server
while True:
      try:
          conn = setup_connection()
          data_transfer(conn, transmitting_data)
      except:
          break


