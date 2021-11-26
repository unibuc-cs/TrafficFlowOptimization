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
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from ray.tune.registry import register_env
from gym import spaces
import numpy as np
import sumo_rl
from sumo_rl import env, parallel_env
import traci
import supersuit as supersuit
import argparse
from datetime import datetime
import massa_common

if __name__ == '__main__':

    args = massa_common.parseArgs()

    assert args.singleagent == 0, "We are expecting to run multiagent for now using RLLIB. if single, use the other file with baselines3"
    ray.init()

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
                              fixed_ts=False)
    # Now pad observations and action spaces to have the same length for observations and actions
    testEnv = supersuit.pad_observations_v0(testEnv)
    testEnv = supersuit.pad_action_space_v0(testEnv)

    testEnv = ParallelPettingZooEnv(testEnv)
    #-------------------------------------------------------


    padded_action_space = testEnv.action_space
    padded_observation_space = testEnv.observation_space

    register_env("MASSA_multiAgentEnv", lambda _: testEnv)

    # TODO: use other strategies and parameters here to leverage policies etc.
    if True:
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

    while True:
        print(trainer.train())  # distributed training step

