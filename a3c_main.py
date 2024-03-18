import argparse
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.multiprocessing as mp

from models import A3C_LSTM_GA
from a3c_train import train_curriculum, train
from a3c_test import test
import logging

from dynamics import *
from distutils.util import strtobool
import numpy as np

import instruction_processing as ip
from progressbar import progressbar


parser = argparse.ArgumentParser(description="Gated-Attention for Grounding")

# Environment arguments
parser.add_argument(
    "-l",
    "--max-episode-length",
    type=int,
    default=2000, # TODO experiment a bit with this value
    help="maximum length of an episode (default: 1000)",
)
parser.add_argument(
    "--living-reward",
    type=float,
    default=0,
    help="""Default reward at each time step (default: 0,
                    change to -0.005 to encourage shorter paths)""",
)
parser.add_argument(
    "-v",
    "--visualize",
    type=int,
    default=0,
    help="""Visualize the envrionment (default: 0,
                    use 0 for faster training)""",
)
parser.add_argument(
    "--sleep",
    type=float,
    default=0,
    help="""Sleep between frames for better
                    visualization (default: 0)""",
)

# A3C arguments
parser.add_argument(
    "--lr",
    type=float,
    default=0.0006,
    metavar="LR",
    help="learning rate (default: 0.001)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.9996,
    metavar="G",
    help="discount factor for rewards (default: 0.9995)",
)
parser.add_argument(
    "--tau",
    type=float,
    default=1.00,
    metavar="T",
    help="parameter for GAE (default: 1.00)",
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "-n",
    "--num-processes",
    type=int,
    default=8,
    metavar="N",
    help="how many training processes to use (default: 4)",
)
parser.add_argument(
    "--num-steps",
    type=int,
    default=600,
    metavar="NS",
    help="number of forward steps in A3C (default: 20)",
)
parser.add_argument(
    "--load",
    type=str,
    default="0",
    help="model path to load, 0 to not reload (default: 0)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    type=int,
    default=0,
    help="""0:Train, 1:Evaluate MultiTask Generalization
                    2:Evaluate Zero-shot Generalization (default: 0)""",
)
parser.add_argument(
    "--dump-location",
    type=str,
    default="./saved/",  # TODO separate into /logs and /saved_models
    help="path to dump models and log (default: ./saved/)",
)

if __name__ == "__main__":
    args = parser.parse_args()
    curriculum_dir_path = Path.cwd() / "data" / "curriculum"
    if not curriculum_dir_path.exists():
        print("No curriculum found - check if curriculum has been generated correctly")
    curriculum_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path.cwd() / "data" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file_path = checkpoint_dir / "model_checkpoint.pth"
    test_dir_path = Path.cwd() / "data" / "test-set" / "level0"
    test_dir_path.mkdir(parents=True, exist_ok=True)

    word_to_idx = ip.get_word_to_idx_from_curriculum_dir(
        curriculum_dir_path=curriculum_dir_path
    )
    args.input_size = len(word_to_idx)

    # debugging
    #args.num_processes = 8

    agents = ["agent/"]

    config_dict = {
        "xmlPath": "", # set in train() / test()
        "infoJson": "",
        "agents": agents,
        "rewardFunctions": [target_reward, distractor_reward, collision_reward], 
        "doneFunctions": [target_done, distractor_done, border_done],
        "skipFrames": 5,
        "environmentDynamics": [Reward],
        "freeJoint": True,
        "renderMode": False,
        "maxSteps":  args.max_episode_length * args.num_processes, 
        "agentCameras": True,
        "tensorboard_writer": None,
        "sensorResolution": (300, 300), # NOTE may require half resolution on apple devices
    }

    if args.evaluate == 0:
        args.use_train_instructions = 1
        log_filename = "train.log"
        use_test = False
    elif args.evaluate == 1:
        args.use_train_instructions = 1 # TODO remove obsolete
        args.num_processes = 0
        curriculum_dir_path = Path.cwd() / "data" / "test-set" 
        curriculum_dir_path.mkdir(parents=True, exist_ok=True)
        log_filename = "test-MT.log"
        use_test = True
    elif args.evaluate == 2:
        args.use_train_instructions = 0
        args.num_processes = 0
        curriculum_dir_path = Path.cwd() / "data" / "test-set" 
        curriculum_dir_path.mkdir(parents=True, exist_ok=True)
        log_filename = "test-ZSL.log"
        use_test = True
    else:
        assert False, "Invalid evaluation type"

    # Setup logging
    dump_location = Path(args.dump_location)
    if not dump_location.exists():
        dump_location.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(dump_location / log_filename), level=logging.INFO)

    device = torch.device("cpu")
    print(f"Shared model using device: {device}")
    shared_model = A3C_LSTM_GA(args, device)
    shared_model.to(device)

    # Load the model
    if args.load != "0":
        shared_model.load_state_dict(
            torch.load(args.load, map_location=lambda storage, loc: storage)
        )

    shared_model.share_memory()

    processes = []

    if use_test:
        p = mp.Process(target=test, args=(args.num_processes, args, shared_model, config_dict, curriculum_dir_path, test_dir_path, checkpoint_file_path, device))
        p.start()
        processes.append(p)
    else:
        for rank in range(args.num_processes):
            p = mp.Process(target=train_curriculum, args=(curriculum_dir_path, rank, args, shared_model, config_dict, checkpoint_file_path, device))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
