import torch
import pfrl.initializers
import torch
import torch.nn as nn

from pfrl import experiments
class AbstractAgent(object):
    def __init__(self):
        # define and create the device
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        self.device = torch.device(device)

    # Act function
    def act(self, observation):
        raise NotImplementedError

    # Observation, gym compatible
    def observe(self, observation, reward, done, info):
        raise NotImplementedError


# This is the type of agent in multi RL problem def where there is a single controlling agent per traffic light
class SingleTrafficLightAgent(AbstractAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__()
        self.config = config

        # The agents available
        self.agents = dict()

    # The observation is a dictionary from agentId -> obs
    def act(self, observation):
        actPerAgent = dict()
        for agId in observation.keys():
            actPerAgent[agId] = self.agents[agId].act(observation[agId])
        return actPerAgent

    def observe(self, observation, reward, done, info):
        for agId in observation.keys():
            self.agents[agId].observe(observation[agId], reward[agId], done, info)
            if done:
                # Save some data from time to time from agents
                if info['eps'] % 200 == 0:
                    self.agents[agId].save(self.config['log_dir']+'agent_'+agId)


# This is the type of agents shared by all traffic signals, used typically in the single RL problem def
class SharedTrafficLightAgent(AbstractAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__()
        self.config = config
        self.agent = None
        self.validActions = None # Signal traffic ID => action
        self.inverseOfValid = None # inverse of above

    def act(self, observation):
        if self.inverseOfValid is None and self.validActions is not None:
            self.inverseOfValid = dict()
            for signal_id in self.validActions:
                self.inverseOfValid[signal_id] = {v: k for k, v in self.validActions[signal_id].items()}

        obs_batch = [observation[agId] for agId in observation.keys()]
        if self.validActions is None:
            valid_reverse = None
            reverse_batch = None
        else:
            valid_reverse = [self.validActions.get(agId) for agId in
                             observation.keys()]
            reverse_batch = [self.inverseOfValid.get(agId) for agId in
                             observation.keys()]

        # Create the batch of actions
        actions_batch = self.agent.act(obs_batch,
                                    valid_acts=valid_reverse,
                                    reverse_valid=reverse_batch)

        # Final result of what each agent should do
        actionsPerAgent = dict()
        for i, agId in enumerate(observation.keys()):
            actionsPerAgent[agId] = actions_batch[i]
        return actionsPerAgent

    def observe(self, observation, reward, done, info):
        # Create batch for observation from all shared agents
        obs_batch = [observation[agId] for agId in observation.keys()]
        rew_batch = [reward[agId] for agId in observation.keys()]
        done_batch = [done]*len(obs_batch)
        reset_batch = [False]*len(obs_batch)

        # Send the observation as batches
        self.agent.observe(obs_batch, rew_batch, done_batch, reset_batch)

        if done:
            if info['eps'] % 200 == 0:
                self.agent.save(self.config['log_dir']+'agent')


def lecunnInitMethod(layer, gain=1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer


