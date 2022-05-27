import os
import sys

import supersuit as supersuit

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import pandas as pd
import ray
from ray.rllib.agents.a3c.a3c import A3CTrainer
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.dqn.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.agents.sac.sac import SACTrainer
from ray.rllib.agents.sac.sac_tf_policy import SACTFPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.contrib.maddpg.maddpg import MADDPGTrainer
from ray.rllib.contrib.maddpg.maddpg_policy import MADDPGTFPolicy
from ray.rllib.agents.sac.sac_tf_policy import SACTFPolicy
from ray.rllib.agents.sac.sac import SACTrainer
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from ray.tune.registry import register_env
from gym import spaces
import numpy as np

import sys
sys.path.append("./sumo-rl")
import sumo_rl
from sumo_rl import env, parallel_env
import traci
import supersuit as supersuit
import argparse
from datetime import datetime
import time
import massa_common

class EnvConfigObj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def debugPrint(self):
        print(self.__dict__)

from ray.tune import Stopper

class TimeStopper(Stopper):
    def __init__(self):
        self._start = time.time()
        self._deadline = 300

    def __call__(self, trial_id, result):
        return False

    def stop_all(self):
        return time.time() - self._start > self.deadline

if __name__ == '__main__':

    args = massa_common.parseArgs()

    assert args.singleagent == 0, "We are expecting to run multiagent for now using RLLIB. if single, use the other file with baselines3"
    ray.init()

    def environmentBuildFunc(env_config):
        print(f"$$$$$ Building for worker index {env_config.worker_index}")
        # Create the environment for the multi agent case
        if False:
            #-------------------------------------------------------
            testEnv = sumo_rl.env(net_file=args.net,
                                  route_file=args.route,
                                  out_csv_name=args.outputbasepath,
                                  use_gui=bool(args.gui),
                                  single_agent=bool(args.singleagent),
                                  max_depart_delay=0,
                                  time_to_teleport=-1, # The time to teleport a vehicle if waiting too much in a queue or something
                                  delta_time = args.delta_time,
                                  yellow_time=args.yellowtime,
                                  min_green=args.mingreen,
                                  max_green=args.maxgreen,
                                  begin_time=args.begin_time,
                                  num_seconds=args.simulation_time,
                                  sumo_seed = args.sumo_seed,
                                  forced_label=env_config.worker_index,
                                  fixed_ts=False)


            # Then finally call the petting zoo env
            testEnv = PettingZooEnv(testEnv)
        else:
            testEnv = parallel_env(net_file=args.net,
                                  route_file=args.route,
                                  out_csv_name=args.outputbasepath,
                                  use_gui=bool(args.gui),
                                  single_agent=bool(args.singleagent),
                                  max_depart_delay=0,
                                  time_to_teleport=-1, # The time to teleport a vehicle if waiting too much in a queue or something
                                  delta_time = args.delta_time,
                                  yellow_time=args.yellowtime,
                                  min_green=args.mingreen,
                                  max_green=args.maxgreen,
                                  begin_time=args.begin_time,
                                  num_seconds=args.simulation_time,
                                  sumo_seed = args.sumo_seed,
                                  forced_label=env_config.worker_index,
                                  fixed_ts=False)

        # Now pad observations and action spaces to have the same length for observations and actions
        testEnv = supersuit.pad_observations_v0(testEnv)
        testEnv = supersuit.pad_action_space_v0(testEnv)
        testEnv = ParallelPettingZooEnv(testEnv)
        return testEnv

    configDemo = {"worker_index" : -1}
    envDemoConfig = EnvConfigObj(**configDemo)
    testEnv = environmentBuildFunc(envDemoConfig)

    padded_action_space = testEnv.action_space
    padded_observation_space = testEnv.observation_space

    register_env("MASSA_multiAgentEnv", lambda config : environmentBuildFunc(config))

    # TODO: use other strategies and parameters here to leverage policies etc.
    if True:
        trainer = SACTrainer(env="MASSA_multiAgentEnv", config={
            "multiagent": {
                "policies": {
                    '0': (SACTFPolicy, padded_observation_space, padded_action_space, {}),
                },
                "policy_mapping_fn": (lambda id: '0')  # Traffic lights are always controlled by this policy
            },
            "lr": args.learningrate,
            "no_done_at_end": True,
        })
    elif True:
        trainer = MADDPGTrainer(env="MASSA_multiAgentEnv", config={
            "multiagent": {
                "policies": {
                    '0': (MADDPGTFPolicy, padded_observation_space, padded_action_space, {'agent_id':0}),
                },
                "policy_mapping_fn": (lambda id: '0')  # Traffic lights are always controlled by this policy
            },
            "lr": args.learningrate,
            "no_done_at_end": True
        })
    elif False:
        trainer = PPOTrainer(env="MASSA_multiAgentEnv", config={
            "multiagent": {
                "policies": {
                    '0': (PPOTFPolicy, padded_observation_space, padded_action_space, {}),
                },
                "policy_mapping_fn": (lambda id: '0')  # Traffic lights are always controlled by this policy
            },
            "lr": args.learningrate,
            "no_done_at_end": False,
        })
    elif False:
        trainer = A3CTrainer(env="MASSA_multiAgentEnv", config={
            "multiagent": {
                "policies": {
                    '0': (A3CTFPolicy, padded_observation_space, padded_action_space, {})
                },
                "policy_mapping_fn": (lambda id: '0')  # Traffic lights are always controlled by this policy
            },
            "lr": args.learningrate,
            "no_done_at_end": True,
        })
    else:
        trainer = DQNTrainer(env="MASSA_multiAgentEnv", config={
            "multiagent": {
                "policies": {
                    '0': (DQNTFPolicy, padded_observation_space, padded_action_space, {})
                },
                "policy_mapping_fn": (lambda id: '0')  # Traffic lights are always controlled by this policy
            },
            "lr": args.learningrate,
            "no_done_at_end": True,
        })

    start = time.time()
    while True:
        result = trainer.train()
        print(result)
        # stop training of the target train steps or reward are reached
        if result["timesteps_total"] >= args.stop_timesteps or (time.time() - start > args.maxTime):
            break

