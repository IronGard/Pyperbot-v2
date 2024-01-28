import configparser
import argparse

#===================Initialise Arg Parser=======================

#initialise parser
parser = argparse.ArgumentParser(description = 'Run python file with or without arguments')

#add arguments
parser.add_argument('-f', '--filename', help = "Name of python file to run")
parser.add_argument('-a', '--algo', help = "Name of algorithm to be run")
parser.add_argument('-en', '--env', help = "Simulation environment to be used")
parser.add_argument('-ep', '--epochs', help = "Number of training iterations")
args = vars(parser.parse_args())

def main():
    #open and read config parser
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'filename': 'main.py', 
                         'epochs': 20
                         }


    #check if user entered necessary parameters:
    config['USER'] = {k: v for k,v in args.items() if v is not None}
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
