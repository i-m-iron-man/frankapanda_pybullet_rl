import pybullet as p
import numpy as np
import time

class Panda:
    def __init__(self, stepsize=1e-3, realtime=0, render= True):
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime

        self.control_mode = "torque" 

        self.position_control_gain_p = [0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        self.position_control_gain_d = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        

        # connect pybullet
        if render:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])
        else:
             p.connect(p.DIRECT)

        p.resetSimulation()
        p.setTimeStep(self.stepsize)
        p.setRealTimeSimulation(self.realtime)
        p.setGravity(0,0,0)

        # load models
        p.setAdditionalSearchPath("models")

        self.plane = p.loadURDF("plane/plane.urdf",
                                useFixedBase=True)
        p.changeDynamics(self.plane,-1,restitution=.95)

        self.robot = p.loadURDF("panda/panda.urdf",
                                useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION)
        self.target = self.spawn_target()
        
        # robot parameters
        self.dof = p.getNumJoints(self.robot) - 1 # Virtual fixed joint between the flange and last link
        if self.dof != 7:
            raise Exception('wrong urdf file: number of joints is not 7')

        self.joints = []
        self.q_min = []
        self.q_max = []
        self.max_torque=[]
        self.target_pos = []
        self.target_torque = []

        for j in range(self.dof):
            joint_info = p.getJointInfo(self.robot, j)
            self.joints.append(j)
            self.q_min.append(joint_info[8])
            self.q_max.append(joint_info[9])
            self.max_torque.append(joint_info[10])
            self.target_pos.append((self.q_min[j] + self.q_max[j])/2.0)
            self.target_torque.append(0.)

        self.reset()

    def reset(self, reset_pos=None):
        self.t = 0.0        
        self.control_mode = "torque"

        if reset_pos is None:
            for j in range(self.dof):
                self.target_torque[j] = 0.
                p.resetJointState(self.robot,j,targetValue=self.target_pos[j])
        else:
            for j in range(self.dof):
                self.target_torque[j] = 0.
                p.resetJointState(self.robot,j,targetValue=reset_pos[j])

        self.resetController()

    def step(self):
        self.t += self.stepsize
        p.stepSimulation()
        time.sleep(self.stepsize)

    # robot functions
    def resetController(self):
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=[0. for i in range(self.dof)])

    def setControlMode(self, mode):
        if mode == "position":
            self.control_mode = "position"
        elif mode == "torque":
            if self.control_mode != "torque":
                self.resetController()
            self.control_mode = "torque"
        else:
            raise Exception('wrong control mode')

    def setTargetPositions(self, target_pos):
        self.target_pos = target_pos
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.target_pos,
                                    forces=self.max_torque,
                                    positionGains=self.position_control_gain_p,
                                    velocityGains=self.position_control_gain_d)

    def setTargetTorques(self, target_torque):
        self.target_torque = target_torque
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=self.target_torque)

    def get_state(self):
        joint_states = p.getJointStates(self.robot,self.joints)
        joint_angs = np.array([x[0] for x in joint_states])
        joint_vels = np.array([x[1]for x in joint_states])
        joint_tors = np.array(self.target_torque)#[x[3] for x in joint_states]
        
        #joint_sens = np.array([x[2] for x in joint_states])
        #joint_sens = joint_sens.flatten()
        
        ee_pose_state = p.getLinkState(self.robot, 6)
        ee_pos = np.array(ee_pose_state[0])
        ee_ori = np.array(ee_pose_state[1])

        ee_vel_state = p.getLinkState(self.robot, 6, computeLinkVelocity=1)
        ee_vel = np.array(ee_vel_state[6])
        ee_ang_vel = np.array(ee_vel_state[7])

        target_pos = np.array(p.getBasePositionAndOrientation(self.target)[0])

        return np.concatenate((joint_angs,joint_vels,joint_tors,ee_pos,ee_ori,ee_vel,ee_ang_vel,target_pos))

    def calc_dist(self,state):
        ee_pos = state[21:24]
        target_pos = state[-3:]
        print("ee_pos: ", ee_pos)
        print("tar_pos: ", target_pos)
        dist = ee_pos-target_pos
        dist = np.linalg.norm(dist)
        print("dist: ",dist)

    def spawn_target(self, pos = np.array([0.5,0.0,0.5]), rad = 0.1):

        visual_params = {
            "radius": rad,
            "specularColor": np.array([255,0,0]),
            "rgbaColor": np.array([1,0,0,1]),
        }
        baseVisualShapeIndex = p.createVisualShape(p.GEOM_SPHERE, **visual_params)
        collision=-1
        return p.createMultiBody(baseVisualShapeIndex=baseVisualShapeIndex,
            baseCollisionShapeIndex=collision,
            baseMass=0.1,
            basePosition=pos,)

    def set_target_vel(self, linear_velocity = np.zeros(3)):
        p.resetBaseVelocity(self.target,linearVelocity=linear_velocity)
    
    def set_target_pos(self, pos = np.zeros(3)):
        p.resetBasePositionAndOrientation(self.target, posObj = pos, ornObj = [0.1,0.5,0.3,0.1])

    def end(self):
        p.disconnect()

if __name__=="__main__":
    duration = 30
    stepsize = 1e-3

    robot = Panda(stepsize)
    robot.setControlMode("torque")

    for i in range(int(duration/stepsize)):
        if i%1000 == 0:
            print("Simulation time: {:.3f}".format(robot.t))
 
        if i%10000 == 0:
            robot.reset()
            robot.setControlMode("torque")
            target_torque = [0 for i in range(robot.dof)]
        #robot.set_target_pos(pos=np.random.rand(3))
        state = robot.get_state()
        robot.calc_dist(state)
        print(state)
        target_torque = [0,0,0,10,0,0,0]

        robot.setTargetTorques(target_torque)
        robot.step()


        time.sleep(robot.stepsize)
