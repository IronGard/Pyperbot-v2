import pybullet as p
import pybullet_data
import random

class Loader():
    '''
    For loading structures into the Pybullet environment.

    Attributes:
        video_dir: str
            the save directory of the mp4 training video.
    Methods:
        plane()
            load the base plane.
        maze(maze_dir)
            load the 20x20 maze based on the given directory.
        robot(robot_dir)
            load the robot based on the given directory.
        goal(num_goals)
            load <num_goals> number of goals.
    '''
    def __init__(self, video_dir):
        '''
        Initiate the Pybullet environment and setup save directory for training video.

        Parameters:
            video_dir: str
                the save directory of the mp4 training video.
        '''
        self._physicsClient = p.connect(p.GUI, options = video_dir)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)

    def plane(self):
        '''
        Load the base plane.
        '''
        p.loadURDF("plane.urdf", physicsClientId = self._physicsClient, basePosition = [0, 0, 0])

    def maze(self, maze_dir):
        '''
        Load the 20x20 maze based on the given directory. 

        Parameter:
            maze_dir: str
                the directory of the stl file.
        '''
        maze_scale = [1, 1, 1]
        maze_visual_id = p.createVisualShape(shapeType=p.GEOM_MESH, 
                                             fileName=maze_dir, 
                                             meshScale=maze_scale)
        maze_collision_id = p.createCollisionShape(shapeType=p.GEOM_MESH, 
                                                   fileName=maze_dir, 
                                                   meshScale=maze_scale, 
                                                   flags=1)
        p.createMultiBody(basePosition=[-10, 0, 0], 
                          baseVisualShapeIndex=maze_visual_id,  
                          baseCollisionShapeIndex=maze_collision_id,
                          physicsClientId = self._physicsClient)

    def robot(self, robot_dir):
        '''
        Load the robot based on the given directory.
        Parameter:
            robot_dir: str
                the directory of the xacro file.
        Return:
            robot_id: int
                the unique id of the loaded robot.
        '''
        robot_id = p.loadURDF(robot_dir, 
                           physicsClientId = self._physicsClient, 
                           basePosition = [0.5, 0.5, 0], 
                           globalScaling = 2)
        return robot_id

    def goal(self, num_goals):
        '''
        Load <num_goals> number of goals.

        Parameter:
            num_goals: int
                the total number of goals in the maze.
        '''
        duck_scale = [1, 1, 1]
        shift = [0, -0.02, 0]
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName="duck.obj",
                                            rgbaColor=[1, 1, 1, 1],
                                            specularColor=[0.4, 0.4, 0],
                                            visualFramePosition=shift,
                                            meshScale=duck_scale)

        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName="duck_vhacd.obj",
                                                collisionFramePosition=shift,
                                                meshScale=duck_scale)

        #Generate random coordinates for the goal
        x, y = [], []
        for i in range(num_goals):
            x.append(random.randrange(-9, 9, 2))
            y.append(random.randrange(1, 19, 2))
            p.createMultiBody(baseMass=1,
                            baseInertialFramePosition=[0, 0, 0],
                            baseCollisionShapeIndex=collisionShapeId,
                            baseVisualShapeIndex=visualShapeId,
                            baseOrientation=[0, 45, 45, 0],
                            basePosition=[x[i], y[i], 0])