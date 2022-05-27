# This is the PPO algorithm for traffic flow optimization evaluation, based on the PFRL library
import torch
import torch.nn as nn
import numpy as np
from pfrl.nn import Branched
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead

from pfrl import experiments
from massa_pfrl_agentdef import SharedTrafficLightAgent, AbstractAgent, lecunnInitMethod
from massa_common import getConv2dOutSize

# for the PPO driving agent type in a single-RL problem def
class TrafficLightPPOAgent(AbstractAgent):
    def __init__(self, config, obs_space, act_space):
        super().__init__()

        self.initMethod = lecunnInitMethod
        width = getConv2dOutSize(obs_space[2])
        height = getConv2dOutSize(obs_space[1])

        # Seq model with an initial conv layer followed by FCs
        self.model = nn.Sequential(
            self.initMethod(nn.Conv2d(obs_space[0], 128, kernel_size=(2, 2))), nn.ReLU(),nn.Flatten(),
            self.initMethod(nn.Linear(height*width*128, 128)), nn.ReLU(),
            self.initMethod(nn.Linear(128, 64)), nn.ReLU(),
            self.initMethod(nn.Linear(128, 32)), nn.ReLU(),

            # Branch the model into actor and critic
            Branched(nn.Sequential(self.initMethod(nn.Linear(32, act_space), 1e-2), SoftmaxCategoricalHead()), # First branch - actor one
                    self.initMethod(nn.Linear(32, 1))       # Second branch - critic
                    )
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.5e-4, eps=1e-5)
        self.agent = PPO(self.model, self.optimizer, gpu=self.device.index,
                         phi=lambda x: np.asarray(x, dtype=np.float32),
                         clip_eps=0.1,
                         clip_eps_vf=None,
                         update_interval=2048,
                         minibatch_size=512,
                         epochs=24,
                         standardize_advantages=True,
                         entropy_coef=0.0015,
                         max_grad_norm=0.65)

    def act(self, observation):
        return self.agent.act(observation)

    def observe(self, observation, reward, done, info):
        self.agent.observe(observation, reward, done, False)

    def save(self, path):
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, path+'.pt')


# The proxy to be used from external calls
class TrafficLightAgentPPOProxy(SharedTrafficLightAgent):

    def save(self, path):
        for agId in self.agents:
            torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()},
                    path + 'agId'+ '.pt')

    def __init__(self, config, observationsAndActions):
        super().__init__(config, observationsAndActions)
        self.agents = {}
        # Create one agent per each traffic light signal
        for id in observationsAndActions:
            obs_space = observationsAndActions[id][0]
            act_space = observationsAndActions[id][1]
            self.agents[id] = TrafficLightPPOAgent(config, obs_space, act_space)

