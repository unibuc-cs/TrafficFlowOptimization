import argparse
from datetime import datetime
import os
import sys
sys.path.append("./sumo-rl")

def parseArgs():
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""MultiAgent RL using RLLib""")
    prs.add_argument("-route", dest="route", type=str, default=None, help="Route definition xml file.\n", required=True)
    prs.add_argument("-net", dest="net", type=str, default=None, help="Net xml file\n", required=True)

    prs.add_argument("-begin_time", dest="begin_time", type=int, default=0,
                     help="When the simulation should start (seconds) in the simulator data ? \n", required=True)
    prs.add_argument("-simulation_time", dest="simulation_time", type=int, default=0,
                     help="How much time (in second) should the simulation run ? \n", required=True)
    prs.add_argument("-sumo_seed", dest="sumo_seed", type=str, default='random',
                     help="The simulation seed used inside SUMO. use one if you need deterministic output results. 'random' if you want it to be random\n",
                     required=True)

    prs.add_argument("-yellowtime", dest="yellowtime", type=int, default=2, required=False, help="Time for yellow\n")
    prs.add_argument("-mingreen", dest="mingreen", type=int, default=5, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="maxgreen", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-delta_time", dest="delta_time", type=int, default=5, required=False,
                     help="Time to advance in simulator after each applied action to the environment\n")

    prs.add_argument("-singleagent", dest="singleagent", type=int, default=1,
                     help="If 1 there will be a single controller (agent)"
                          "for all traffic lights. 0 means we have multi agent RL, with an individual agent attached to each traffic light\n")

    prs.add_argument("-outputbasepath", dest="outputbasepath", type=str, default=None,
                     help="Base path to write results on\n", required=True)
    prs.add_argument("-gui", dest="gui", type=int, default=0, help="Should use SUMO GUI ? \n")

    prs.add_argument("-lr", dest="learningrate", type=float, default=0.001, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-stop_timesteps", dest="stop_timesteps", type=int, required=True, help="NUmber of total timesteps to stop at")
    prs.add_argument("-maxTime", dest="maxTime", type=int, required=True,
                     help="Maximum time in seconds to run an experiment")

    args = prs.parse_args()
    experiment_time = str(datetime.now()).split('.')[0]
    #args.outputbasepath = args.outputbasepath + "/" + experiment_time #os.path.join(args.outputbasepath, experiment_time)

    return args