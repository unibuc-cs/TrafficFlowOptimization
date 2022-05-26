import sys
import sumo
import os
from argparse import ArgumentParser
import json
import threading
import subprocess




# Getting the env variable
SUMO_HOME = os.environ.get("SUMO_HOME", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    basestring
    # Allows isinstance(foo, basestring) to work in Python 3
except NameError:
    basestring = str

assert SUMO_HOME is None, "You have to setup SUMO_HOME env var. Please follow SUMO installation notes !"
SUMO_TOOLS = os.environ.join(SUMO_HOME, "tools")
sys.path.append(SUMO_TOOLS)

import osmGet
import osmWebWizard
import osmBuild


# Parse the arguments

parser = ArgumentParser(
    description="Capturing dataset tool for RL simulation. TODO: many customizations are needede for traffic flow generation !")
parser.add_argument("--locationToCapture", default="", help="Location in lat/long for the bounding box to capture from the real world globe", dest="locationToCapture")
parser.add_argument("-d", "--output-directory", default=os.getcwd(),
                       help="directory in which to put the output files", dest="outputDir")

args = parser.parse_args()


# Get the xml with the map at the specified location
osmGet.get(["-b=" + args.locationToCapture, "-d", args.outputDir]) # osmGet.py -b 51.097,17.0192,51.1192,17.0659 -p "bucharest" -d "C:/Test"
subprocess.call([os.path.join(SUMO_TOOLS, "osmWebWizard.py", f"--test-output={args.outputDir}")]) # --osm-file "C:/Test/bucharest_bbox.osm.xml"
subprocess.call([os.path.join(SUMO_TOOLS, "osmBuild.py", f"-d")]) # --osm-file "C:/Test/bucharest_bbox.osm.xml"

