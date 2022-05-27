import gym
from stable_baselines3.dqn.dqn import DQN
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import sys
sys.path.append("./sumo-rl")
from sumo_rl import SumoEnvironment
import traci
import massa_common

if __name__ == '__main__':
    args = massa_common.parseArgs()

    assert args.singleagent == 1, "We are expecting to run single agent for now using baslines. if single, use the other file with baselines3"

    env = SumoEnvironment(net_file=args.net,
                          route_file=args.route,
                          out_csv_name=args.outputbasepath,
                          use_gui=bool(args.gui),
                          single_agent=bool(args.singleagent),
                          max_depart_delay=0,
                          time_to_teleport=-1,
                          # The time to teleport a vehicle if waiting too much in a queue or something
                          delta_time=args.delta_time,
                          yellow_time=args.yellowtime,
                          min_green=args.mingreen,
                          max_green=args.maxgreen,
                          begin_time=args.begin_time,
                          num_seconds=args.simulation_time,
                          sumo_seed=args.sumo_seed,
                          fixed_ts=False)


    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.02,
        learning_starts=0,
        train_freq=1,
        target_update_interval=100,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1
    )
    model.learn(total_timesteps=100000)

