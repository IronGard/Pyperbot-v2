import pybullet as p
import pybullet_data
import os
import random

class UnevenTerrain:
    def __init__(self, seed, num_boxes = 10):
        '''
        Constructor for num boxes in the terrain to create uneven heightfield in the terrain
        '''
        self.num_boxes = num_boxes

    def generate_terrain(self):
        '''
        function to generate terrain with boxes with random position and sizes
        '''
        # Set up the simulation
        physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)

        #load the terrain file
        terrain = p.loadURDF("plane.urdf")

        #add some boxes with random sizes and positions
        for i in range(self.num_boxes):
            #Random position
            x = random.uniform(-2,2)
            y = random.uniform(-2,2)
            z = random.uniform(0,1)

            #Random size
            sx = random.uniform(0.1,0.5)
            sy = random.uniform(0.1,0.5)
            sz = random.uniform(0.1,0.5)
            
            #create collision shape for box
            box_collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx,sy,sz])
            
            #create visual shape for box
            box_visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx,sy,sz], rgbaColor=[1,0,0,1])

            #set position and orientation
            box_position = [x,y,z]
            box_orientation = p.getQuaternionFromEuler([0,0,0])

            #create the box as multi-body with collision and visual shape
            box_id = p.createMultiBody(baseCollisionShapeIndex=box_collision_id,
                                       baseVisualShapeIndex=box_visual_id, 
                                       basePosition=box_position, 
                                       baseOrientation=box_orientation, 
                                       physicsClientId=physicsClient)