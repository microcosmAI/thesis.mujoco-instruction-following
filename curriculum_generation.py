# Script that calls the following scripts in order:
#  1. ./colorset_generation/colorset_writer.py
#  2. ./prompt_generation/prompt_writer.py
#  3. ./xml_generation/xml_writer.py
#  
# This script should be structured as follows:
#  1. Import all necessary modules
#  2. Ask user whether they want default or custom curriculum
#  3. If default, call curriculum generation function with default parameters
#  4. If custom, ask user for parameters:
#     a. Generate colorset?
#      i. If yes, call colorset generation function with user parameters
#     b. Generate prompt?
#      i. If yes, call prompt generation function with user parameters
#     c. Generate XML?
#      i. If yes, call XML generation function with user parameters
#     d. Generate curriculum?
#      i. If yes, call curriculum generation function with user parameters
#     (the steps above should be called such that if one is yes, the ones that follow are also yes)
#  5. Call curriculum generation function with user parameters
#  6. Print confirmation message
#
# The curriculum generation script works as follows:
#  Depending on the parameters, create total_levels amount of folders
#  For each folder, do step 4 from above if the user wants custom parameters, 3 otherwise

import os
import sys
import xml.etree.ElementTree as ET
import argparse



# Import all necessary modules
from colorset_generation import colorset_writer
from prompt_generation import prompt_writer
from xml_generation import xml_writer

# Ask user whether they want default or custom curriculum
# Use argpase for this:
#  -d or --default for default
#  -c or --custom for custom
#  -h or --help for help
parser = argparse.ArgumentParser(description='Generate a curriculum.')
parser.add_argument('-d', '--default', action='store_true', help='Generate a default curriculum.')
parser.add_argument('-c', '--custom', action='store_true', help='Generate a custom curriculum.')

args = parser.parse_args()

# If default, call curriculum generation function with default parameters

