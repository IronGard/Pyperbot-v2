import configparser
import argparse

#===================Initialise Arg Parser=======================


def config_setup():
    #initialise parser
    parser = argparse.ArgumentParser(description = 'Run python file with or without arguments')

    #add arguments
    parser.add_argument('-f', '--filename', help = "Name of python file to run")
    parser.add_argument('-a', '--algo', help = "Name of algorithm to be run")
    parser.add_argument('-en', '--env', help = "Simulation environment to be used")
    parser.add_argument('-t', '--terrain', help = "Terrain to be used for the simulation")
    parser.add_argument('-ep', '--episodes', help = "Number of training iterations")
    args = vars(parser.parse_args())

    #open and read config parser
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'filename': 'main.py', 
                         'episodes': 1000,
                         'algo': 'PPO',
                         'env': 'SnakebotEnv',
                         'terrain': 'Plane'}


    #check if user entered necessary parameters:
    config['USER'] = {k: v for k,v in args.items() if v is not None}
    
    #write entries into the config file
    with open('config.yaml', 'w') as configfile:
        config.write(configfile)
    
    #use default values if others not given.
        
#define function to read config files
def read_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

