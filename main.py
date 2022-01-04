from __future__ import division
from env import Env
import torch
from torch.autograd import Variable
import os
import gc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import ddpg.train as train
import ddpg.buffer as buffer

MAX_EPISODES = 5
MAX_STEPS = 500
MAX_BUFFER = 1000000

agent = Env()
initial_state = agent.episode_reset()
    
state_size = len(initial_state)
action_size = agent.action_size
action_lim = 10.0

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(state_size, action_size, action_lim, ram)

reward_list=[]

for ep in range(MAX_EPISODES):
    state = np.float32(agent.episode_reset())
    trainer.reset()
    total_reward = 0.0
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_steps = MAX_STEPS

    for step in range(MAX_STEPS):
        action = trainer.get_exploration_action(state)
        new_state, reward, done = agent.step(action , state)
        total_reward += reward
        new_state = np.float32(new_state)

        ram.add(state, action, reward, new_state, done)

        state = new_state

        #optimize
        critic_loss , actor_loss = trainer.optimize()
        total_critic_loss += critic_loss
        total_actor_loss += actor_loss

        if done:
            total_steps = step+1
            break

    gc.collect()
    print("ep:",ep," total_reward:",total_reward, " steps:",total_steps)
    print("actor loss: ",total_actor_loss)
    print("critic loss: ",total_critic_loss)
    print("\n")




