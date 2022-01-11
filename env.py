from panda_bullet import Panda
import numpy as np

class Env():
    def __init__(self, render=False):
        self.step_size = 1e-3
        self.robot = Panda(self.step_size, render=render)
        self.robot.setControlMode("torque")
        self.action_size = self.robot.dof
    
    def calc_Dist(self,state):
        dist = state[21:24]-state[-3:]
        dist = np.linalg.norm(dist)
        return dist

    def get_State(self):
        return self.robot.get_state()

    def get_Reward(self, old_dist, new_dist):
        return -0.5*old_dist
    
    def is_Done(self, new_dist):
        return new_dist < 0.1

    def step(self, action, old_state):
        old_dist = self.calc_Dist(old_state)

        #do action
        self.robot.setTargetTorques(action)
        self.robot.step()

        new_state = self.get_State()
        
        new_dist = self.calc_Dist(new_state)
        
        reward = self.get_Reward(old_dist,new_dist)

        done = self.is_Done(new_dist)

        return new_state, reward, done
    
    def episode_reset(self, infer=False):
        if infer:
            self.robot.reset()
        else:
            reset_pos=[]
            for i in range(self.robot.dof):
                reset_pos.append(np.random.uniform(self.robot.q_min[i],self.robot.q_max[i]))
            self.robot.reset(reset_pos)
        #reset target pos
        while True:
            x= np.random.uniform(0.0,1.0)
            y= np.random.uniform(-1.0,1.0)
            z= np.random.uniform(0.0,1.0)
            target_pos = np.array([x,y,z])
            if np.linalg.norm(target_pos) < 1.0:
                break
        self.robot.set_target_pos(pos=target_pos)
        return self.get_State()
