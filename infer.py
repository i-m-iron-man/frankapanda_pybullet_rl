from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
import os
import gc
import argparse
import time
from env import Env

import ddpg.train as train
import ddpg.buffer as buffer

agent = Env(render=True)
state = agent.episode_reset(infer=True)

MAX_BUFFER =10
MAX_timestep=10000

state_size = len(state)
action_size = agent.action_size
action_lim = 10.0

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(state_size, action_size, action_lim, ram)

trainer.load_models(5)
total_reward=0.0
state =np.float32(state)

for step in range(MAX_timestep):
    action = trainer.get_exploitation_action(state)
    new_state, reward, done = agent.step(action , state)
    total_reward += reward
    state = np.float32(new_state)
    if done:
        break

gc.collect()
print("total reward: ",total_reward)


