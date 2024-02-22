import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Snakebot')
    parser.add_argument('-m', '--mode', type = str, default = 'p.GUI', help = 'Mode: p.DIRECT or p.GUI')
    parser.add_argument('-ng', '--num_goals', type = int, default = 3, help = 'Number of goals')
    parser.add_argument('-dt', '--delta_time', type = float, default = 1./240., help = 'Time step')
    parser.add_argument('-sp', '--snake_period', type = float, default = 0.1, help = 'Period of the snake')
    parser.add_argument('-wl', '--waveLength', type = float, default = 0.5, help = 'Wave length of the snake')
    parser.add_argument('-wp', '--wavePeriod', type = float, default = 0.5, help = 'Wave period of the snake')
    parser.add_argument('-wa', '--waveAmplitude', type = float, default = 1, help = 'Amplitude of the snake')
    parser.add_argument('-wf', '--waveFront', type = float, default = 0, help = 'Front of the wave of the snake')
    parser.add_argument('-sl', '--segmentLength', type = int, default = 0.05, help = 'Length of the snake')
    parser.add_argument('-st', '--steering', type = float, default = 0, help = 'Steering of the snake')
    parser.add_argument('-a', '--amp', type = float, default = 0.5, help = 'Amplitude of the snake')
    parser.add_argument('-off', '--offset', type = float, default = 0, help = 'Offset of the snake')
    args = parser.parse_args()
    return args