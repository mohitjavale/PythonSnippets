# %%
import pybullet as p
import pybullet_data as pd
import numpy as np
import time

# %%
class Bullet_sim:

    def __init__(self) -> None:        
        physicsClient = p.connect(p.GUI)
        self.timestep = 1/240
        p.setTimeStep(self.timestep)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # can turn off rendering while loading objects to speed up
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) # turn on rendering once object loaded
        p.setPhysicsEngineParameter(jointFeedbackMode=p.JOINT_FEEDBACK_IN_JOINT_FRAME)
        self.lateral_friction_value = 0.5

    def load_ground(self):
        groundPlane_angle = np.deg2rad([0,30,0])
        groundPlane_id = p.loadURDF("plane.urdf", baseOrientation=p.getQuaternionFromEuler(groundPlane_angle))
        p.changeDynamics(groundPlane_id, -1, lateralFriction=self.lateral_friction_value)

    def load_wedge(self):
        wedge_angle = np.deg2rad([0,-30,0])
        wedge_base_position = np.array([5,0,0])
        wedge_length = 2

        wedge1Shape = p.createCollisionShape(shapeType = p.GEOM_BOX, halfExtents=[wedge_length/np.cos(abs(wedge_angle[1]))/2, 1, 0])
        wedge1_id  = p.createMultiBody(0, wedge1Shape,basePosition=wedge_base_position+np.array([-wedge_length,0,np.tan(abs(wedge_angle[1]))*wedge_length/2]),  baseOrientation=p.getQuaternionFromEuler(wedge_angle))
        p.changeDynamics(wedge1_id, -1, lateralFriction=self.lateral_friction_value)
        wedge2Shape = p.createCollisionShape(shapeType = p.GEOM_BOX, halfExtents=[wedge_length/2, 1, np.tan(abs(wedge_angle[1]))*wedge_length/2])
        wedge2_id  = p.createMultiBody(0, wedge2Shape,basePosition=wedge_base_position+np.array([0,0,np.tan(abs(wedge_angle[1]))*wedge_length/2]),  baseOrientation=p.getQuaternionFromEuler(np.deg2rad([0,-0,0])))
        p.changeDynamics(wedge2_id, -1, lateralFriction=self.lateral_friction_value)
        wedge3Shape = p.createCollisionShape(shapeType = p.GEOM_BOX, halfExtents=[wedge_length/np.cos(abs(wedge_angle[1]))/2, 1, 0])
        wedge3_id  = p.createMultiBody(0, wedge3Shape,basePosition=wedge_base_position+np.array([wedge_length,0,np.tan(abs(wedge_angle[1]))*wedge_length/2]),  baseOrientation=p.getQuaternionFromEuler(-wedge_angle))
        p.changeDynamics(wedge3_id, -1, lateralFriction=self.lateral_friction_value)

    def load_randomCubeTerrain(self):
        for i in range(250):
            wedge3Shape = p.createCollisionShape(shapeType = p.GEOM_BOX, halfExtents=np.random.random(3)/3)
            rand_pos = np.random.random(3)*10
            rand_pos[2] = 0
            wedge3_id  = p.createMultiBody(0, wedge3Shape,basePosition=rand_pos,  baseOrientation=p.getQuaternionFromEuler(np.random.random(3)))
            p.changeDynamics(	wedge3_id, -1, lateralFriction=self.lateral_friction_value)
              

    def load_robot(self):
        startPos = [0,0,1]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.robot_id = p.loadURDF("r2d2.urdf",startPos, startOrientation)
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # print robot joint data
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            print(f"joint index = {joint_info[0]}, joint name = {joint_info[1]}")

        # enable joint reaction forces
        for i in range(p.getNumJoints(self.robot_id)):
            p.enableJointForceTorqueSensor(self.robot_id, i, 1)

    def get_robotJointStates(self):
        joint_states = p.getJointStates(self.robot_id, [x for x in range(self.num_joints)])
        joint_positions = [joint_data[0] for joint_data in joint_states]
        joint_velocities = [joint_data[1] for joint_data in joint_states]
        joint_rxnForces = [joint_data[2] for joint_data in joint_states]
        joint_appliedTorques = [joint_data[3] for joint_data in joint_states]

    def set_robotJointActions(self, actions):
        p.setJointMotorControlArray( self.robot_id, [x for x in range(self.num_joints)], controlMode=p.POSITION_CONTROL, targetPositions=actions, positionGains=[1 for x in range(self.num_joints)], velocityGains=[0.5 for x in range(self.num_joints)])
        # p.setJointMotorControlArray( self.robot_id, [x for x in range(self.num_joints)], controlMode=p.VELOCITY_CONTROL, targetVelocities=actions, velocityGains=0.5)
        # p.setJointMotorControlArray( self.robot_id, [x for x in range(self.num_joints)], controlMode=p.TORQUE_CONTROL, forces=actions)


    def step_sim(self):
            p.stepSimulation()
            time.sleep(self.timestep)

    def reset(self):
        p.resetSimulation()
        self.loadGroundPlane()
        self.loadRobot()

    def cameraFollow(self):
        basePos, baseOrn = p.getBasePositionAndOrientation(self.robotId)  # Get model position
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=75, cameraPitch=-20, cameraTargetPosition=basePos)  # fix camera onto model

        


# %%
if __name__=='__main__':
    # %%
    sim = Bullet_sim()
    sim.load_ground()
    sim.load_robot()
    # sim.load_wedge()
    # sim.load_randomCubeTerrain()
    for i in range(10000):
         sim.set_robotJointActions([0 for i in range(sim.num_joints)])
         sim.step_sim()
# %%
