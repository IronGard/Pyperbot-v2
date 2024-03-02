import argparse

def parse_args():
    '''
    Function to accept arguments for controlling the simulation environment.
    '''
    parser = argparse.ArgumentParser(description='snakebot_sim')
    parser.add_argument('-e', '--env', type=str, default='none', help='Sim env: none, maze, lab.')
    parser.add_argument('-g', '--gait', type=str, default='lateral_undulation', help='snake gaits: lateral undulation, concertina_locomotion')
    parser.add_argument('-ng', '--num_goals', type = int, default = 3, help = 'Number of goals (maze env only).')
    args = parser.parse_args()

    return args