class Info():
    def __init__(self):
        
    def joint_classification(robot):
        revolute_joints = []
        prismatic_joints = []
        fixed_joints = []
        other_joints = []
        for i in range(p.getNumJoints(robot)):
            joint_info = p.getJointInfo(robot, i)
            if joint_info[2] == 0:
                revolute_joints.append(i)
            elif joint_info[2] == 1:
                prismatic_joints.append(i)
            elif joint_info[2] == 4:
                fixed_joints.append(i)
            else:
                other_joints.append(i)
        print("revolute joints: ", revolute_joints)
        print("prismatic joints: ", prismatic_joints)
        print("fixed joints: ", fixed_joints)
        print("other joints", other_joints)