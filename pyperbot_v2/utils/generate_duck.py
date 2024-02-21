import pybullet as p
import pybullet_data
import random

#setting up the pybullet environment
physicsClient = p.connect(p.GUI, options = "--mp4=results/videos/training_video.mp4") #save the training video to an mp4 video
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)

#loading the plane
planeID = p.loadURDF("plane.urdf", 
                     physicsClientId = physicsClient, 
                     basePosition = [0, 0, 0])

#loading the maze
maze_dir = "pyperbot_v2/snakebot_description/meshes/maze_10x10.stl"
maze_scale = [1, 1, 1]
maze_visual_id = p.createVisualShape(shapeType=p.GEOM_MESH, 
                                     fileName=maze_dir, 
                                     meshScale=maze_scale)
maze_collision_id = p.createCollisionShape(shapeType=p.GEOM_MESH, 
                                           fileName=maze_dir, 
                                           meshScale=maze_scale, 
                                           flags=1)
maze_body_id = p.createMultiBody(basePosition=[-10, 0, 0], 
                                 baseVisualShapeIndex=maze_visual_id, baseCollisionShapeIndex=maze_collision_id)

#loading snake robot into the environment
robot = p.loadURDF("pyperbot_v2/snakebot_description/urdf/updated_snakebot.urdf.xacro", 
                   physicsClientId = physicsClient, 
                   basePosition = [0.5, 0.5, 0], 
                   globalScaling = 2)

#loading the goal (duck) into the environment
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
num_goal = 5
x, y = [], []
for i in range(num_goal):
    x[i] = random.randrange(-9, 9, 2)
    y[i] = random.randrange(1, 19, 2)
    p.createMultiBody(baseMass=1,
                    baseInertialFramePosition=[0, 0, 0],
                    baseCollisionShapeIndex=collisionShapeId,
                    baseVisualShapeIndex=visualShapeId,
                    baseOrientation=[0, 45, 45, 0],
                    basePosition=[x[i], y[i], 0])