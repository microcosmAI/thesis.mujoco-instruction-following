import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.multiprocessing as mp

from models import A3C_LSTM_GA
from a3c_train import train
#from a3c_test import test

import logging

from MuJoCo_Gym.mujoco_rl import MuJoCoRL
from MuJoCo_Gym.wrappers import GymnasiumWrapper, GymWrapper
from gymnasium.wrappers.frame_stack import FrameStack

# from gymnasium.wrappers import NormalizeObservationV0
from dynamics import *
import argparse
import os
import random
import time
from distutils.util import strtobool
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# from wrappers.record_episode_statistics import RecordEpisodeStatistics
from progressbar import progressbar


def make_env(config_dict):
    def thunk():
        env = MuJoCoRL(config_dict=config_dict)
        env = GymWrapper(env, config_dict["agents"][0])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.action_space.seed(1)
        env.observation_space.seed(1)

        return env

    return thunk

parser = argparse.ArgumentParser(description='Gated-Attention for Grounding')

# Environment arguments
parser.add_argument('-l', '--max-episode-length', type=int, default=30,
                    help='maximum length of an episode (default: 30)')
parser.add_argument('--living-reward', type=float, default=0,
                    help="""Default reward at each time step (default: 0,
                    change to -0.005 to encourage shorter paths)""")
parser.add_argument('--frame-width', type=int, default=300,
                    help='Frame width (default: 300)')
parser.add_argument('--frame-height', type=int, default=168,
                    help='Frame height (default: 168)')
parser.add_argument('-v', '--visualize', type=int, default=0,
                    help="""Visualize the envrionment (default: 0,
                    use 0 for faster training)""")
parser.add_argument('--sleep', type=float, default=0,
                    help="""Sleep between frames for better
                    visualization (default: 0)""")
parser.add_argument('--all-instr-file', type=str, # TODO adapt to new dataset
                    default="data/instructions_all.json",
                    help="""All instructions file
                    (default: data/instructions_all.json)""")
parser.add_argument('--train-instr-file', type=str,
                    default="data/instructions_train.json",
                    help="""Train instructions file
                    (default: data/instructions_train.json)""")
parser.add_argument('--test-instr-file', type=str,
                    default="data/instructions_test.json",
                    help="""Test instructions file
                    (default: data/instructions_test.json)""")
parser.add_argument('--object-size-file', type=str, # TODO make sure size modification gets added to xml generation
                    default="data/object_sizes.txt",
                    help='Object size file (default: data/object_sizes.txt)')

# A3C arguments
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('-n', '--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--load', type=str, default="0",
                    help='model path to load, 0 to not reload (default: 0)')
parser.add_argument('-e', '--evaluate', type=int, default=0,
                    help="""0:Train, 1:Evaluate MultiTask Generalization
                    2:Evaluate Zero-shot Generalization (default: 0)""")
parser.add_argument('--dump-location', type=str, default="./saved/", # TODO separate into /logs and /saved_models
                    help='path to dump models and log (default: ./saved/)')

if __name__ == "__main__":
    
    args = parser.parse_args()

    if args.evaluate == 0:
        args.use_train_instructions = 1
        log_filename = "train.log"
    elif args.evaluate == 1:
        args.use_train_instructions = 1
        args.num_processes = 0
        log_filename = "test-MT.log"
    elif args.evaluate == 2:
        args.use_train_instructions = 0
        args.num_processes = 0
        log_filename = "test-ZSL.log"
    else:
        assert False, "Invalid evaluation type"

    #env = grounding_env.GroundingEnv(args)
    #args.input_size = len(env.word_to_idx)
    args.input_size = 10 # TODO remove

    # set paths and such for the config dict
    # path to folder xml_files from current dir:
    xml_file_path = os.path.join(
        os.getcwd(), "xml_debug_files", "advance_to_the_tea_tree.xml"
    )
    json_files = os.path.join(
        os.getcwd(), "xml_debug_files", "advance_to_the_tea_tree.json"
    )
    agents = ["agent/"]
    num_envs = 2

    config_dict = {
        "xmlPath": xml_file_path,
        "infoJson": json_files,
        "agents": agents,
        "rewardFunctions": [target_reward],  # add collision reward later
        "doneFunctions": [target_done],
        "skipFrames": 5,
        "environmentDynamics": [Reward],
        "freeJoint": True,
        "renderMode": False,
        "maxSteps": 4096 * 16,
        "agentCameras": True,
        "tensorboard_writer": None,
        "sensorResolution": (300, 168),
    }



    # Setup logging
    if not os.path.exists(args.dump_location):
        os.makedirs(args.dump_location)
    logging.basicConfig(filename=args.dump_location+log_filename,
                        level=logging.INFO)
    
    shared_model = A3C_LSTM_GA(args)

    # Load the model
    if (args.load != "0"):
        shared_model.load_state_dict(
            torch.load(args.load, map_location=lambda storage, loc: storage))

    shared_model.share_memory()

    processes = []

    # Start the test thread
    #p = mp.Process(target=test, args=(args.num_processes, args, shared_model))
    #p.start()
    #processes.append(p)

    # Debugging: start a single training thread
    print("Starting a single training thread")
    train(0, args, shared_model, config_dict)
    print("Finished training")

    # Start the training thread(s)
    for rank in range(0, args.num_processes):
        print("Starting training thread", rank)
        #p = mp.Process(target=train, args=(rank, args, shared_model, config_dict))
        #p.start()
        #processes.append(p)
        train(rank, args, shared_model, config_dict)
    #for p in processes:
        #p.join()




