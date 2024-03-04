import socket
import numpy as np
import csv


#Class for creating server object
class server():
    def __init__(self, csv_file_path):
        self.host = '0.0.0.0'
        self.port = 6969
        self.server = self.setup_server()
        self.conn = self.setup_connection()

        #Received message
        self.received_command = ""
        self.received_data = ""
        self.received_message_status = "UNREAD"

        #Transmitting message
        self.csv_file_path = csv_file_path
        self.csv_data = self.read_file()
        self.transmitting_command = "MOVE"
        self.transmitting_data = ""

        #Attributes for reading csv file (to be removed later)
        self.csv_file_path = csv_file_path
        self.dummy_transmitting_data = self.read_file()
        self.n = 0

    #Function for reading joint angles from the csv file
    def read_file(self):
        with open(self.csv_file_path, "r") as file:
            csv_reader = csv.reader(file)
            sample_number = 0
            sample_interval = 10
            starting_point = 50
            transmitting_data = []

            for line in csv_reader:
                if (sample_number >= starting_point):
                    if (sample_number % sample_interval == 0):
                        read_line = line
                        send_line = ""

                        for joint_number in range(20):
                            rounded_read_line = str(round(float(read_line[joint_number]), 2))  # Rounds to 2dp
                            send_line = send_line + rounded_read_line

                            if (joint_number < 19):
                                send_line = send_line + ","

                        transmitting_data.append(send_line)
                sample_number = sample_number + 1
        return transmitting_data

    #Function for returning the n^th sample from transmitting_data
    def feed_transmitting_data(self, fed_data):
        self.transmitting_data = fed_data

    def get_transmitting_data(self):
        return self.transmitting_data

    #Function for setting up the server. Returns a socket object, which is used to form a connection between host and client.
    def setup_server(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket has been created.")
        try:
            s.bind((self.host, self.port))
        except socket.error as msg:
            print(msg)

        print("Socket bind complete")

        return s

    #Function for setting up connection. Returns the comms channel for the comms between server and client.
    def setup_connection(self):
        #Server only listens to 1 device
        self.server.listen(1)
        conn, address = self.server.accept()
        print("Connected to: " + address[0] + ": " + str(address[1]))
        return conn

    #Function for receiving message from the client
    def receive_message(self):
        received_message = self.conn.recv(1024)

        #Splits message into three parts: [COMMAND DATA MESSAGE_STATUS]
        print(received_message)
        received_message = received_message.decode('utf-8')
        received_message = received_message.split(" ")
        self.received_command = received_message[0]
        self.received_data = received_message[1]
        self.received_message_status = received_message[2]

    #Function for returning message_status. Checks if the received message has been executed.
    def get_message_status(self):
        return self.received_message_status

    #Function for executing the received message
    def execute_message(self):
        #Updates the message status to "READ", i.e., the message has been executed.
        self.received_message_status = "READ"

        #This is when the snake requests the next set of joint angles. Server will compute next set of angles and send them back to snake.
        if(self.received_command == "REQ_JOINT_POS"):
            print("Returning new joint positions.")
            print("Sensor data is: " + self.received_data)
            self.transmitting_command = "MOVE_JOINT"
            self.transmitting_data = self.csv_data[self.n]
            self.n = self.n + 1

        #This is an ACK from the robot that it has moved to the commanded joint values
        if(self.received_command == "JOINT_MOVED"):

            #So now the server will request the current sensor data so that it can compute the next set of joint values
            print("Requesting sensor data.")
            self.transmitting_command = "REQ_SENSOR_DATA"
            self.transmitting_data = "NULL"

        #Concats the COMMAND, DATA, and MESSAGE_STATUS, encodes it, and transmits it to the snake
        transmitting_message = self.transmitting_command + " " + self.transmitting_data + " " + "UNREAD"
        self.conn.sendall(str.encode(transmitting_message))
        print(transmitting_message)

    #Function for debugging the snake (test if the joints work) by setting all joints to their initial positions
    def debug(self):
        self.transmitting_command = "DEBUG"
        self.transmitting_data = "NULL"
        transmitting_message = self.transmitting_command + " " + self.transmitting_data + " " + "UNREAD"
        print("Debug")
        print(transmitting_message)
        self.conn.sendall(str.encode(transmitting_message))