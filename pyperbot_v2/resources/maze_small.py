#Define code to generate a maze from the image
import pybullet as p
import cv2
import numpy as np
import pybullet_data

# Connect to the PyBullet server
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

def convert_image_to_urdf(image_path, output_urdf_path):
    # Read the image
    '''
    Function to create custom maze from image by reading contours
    '''
    maze_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Perform image processing (e.g., thresholding) to extract maze information
    _, maze_binary = cv2.threshold(maze_image, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(maze_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Generate URDF file
    with open(output_urdf_path, 'w') as urdf_file:
        urdf_file.write('<?xml version="1.0" ?>\n')
        urdf_file.write('<robot name="maze_robot">\n')

        for idx, contour in enumerate(contours):
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Create a link for each wall
            urdf_file.write(f'  <link name="wall{idx + 1}_link">\n')
            urdf_file.write('    <!-- Visual representation -->\n')
            urdf_file.write(f'    <visual>\n')
            urdf_file.write(f'      <geometry>\n')
            urdf_file.write(f'        <box size="{w * 0.01} {h * 0.01} 0.1" />\n')  # Adjust scale as needed
            urdf_file.write(f'      </geometry>\n')
            urdf_file.write(f'    </visual>\n')
            urdf_file.write('    <!-- Collision representation -->\n')
            urdf_file.write(f'    <collision>\n')
            urdf_file.write(f'      <geometry>\n')
            urdf_file.write(f'        <box size="{w * 0.01} {h * 0.01} 0.1" />\n')  # Adjust scale as needed
            urdf_file.write(f'      </geometry>\n')
            urdf_file.write(f'    </collision>\n')
            urdf_file.write('  </link>\n')

            # Create a joint to connect the wall to the world
            urdf_file.write(f'  <joint name="world_wall{idx + 1}_joint" type="fixed">\n')
            urdf_file.write(f'    <parent link="world" />\n')
            urdf_file.write(f'    <child link="wall{idx + 1}_link" />\n')
            urdf_file.write(f'    <origin xyz="{x * 0.01} {y * 0.01} 0" rpy="0 0 0" />\n')  # Adjust scale as needed
            urdf_file.write(f'  </joint>\n')

        urdf_file.write('</robot>\n')

class MazeSmall:
    '''
    Maze Loader for the maze into snakebot test environment.
    '''
    def __init__(self, client):
        self._client = client
        self.maze = self.gen_maze()
        
    def gen_maze(self, maze_dir):
        '''
        Function to generate the maze in the environment
        '''
        maze_scale = [1, 1, 1]
        maze_visual_id = p.createVisualShape(shapeType=p.GEOM_MESH, 
                                             fileName=maze_dir, 
                                             meshScale=maze_scale)
        maze_collision_id = p.createCollisionShape(shapeType=p.GEOM_MESH, 
                                                   fileName=maze_dir, 
                                                   meshScale=maze_scale, 
                                                   flags=1)
        maze = p.createMultiBody(basePosition=[0, 0, 0], 
                          baseVisualShapeIndex=maze_visual_id,  
                          baseCollisionShapeIndex=maze_collision_id,
                          physicsClientId = self._physicsClient)
        return maze

if __name__ == "__main__":
    image_path = "path/to/maze_image.png"
    output_urdf_path = "output/maze.urdf"
    
    convert_image_to_urdf(image_path, output_urdf_path)

    #create pybullet simulation and try to load the urdf to see if it can be visualised