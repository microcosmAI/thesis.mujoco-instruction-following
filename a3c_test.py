#import numpy as np
#import torch
#import torch.nn.functional as F
import time
#import logging
##
#import env as grounding_env
#from models import A3C_LSTM_GA

#from torch.autograd import Variable
#from constants import *

from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import logging

from MuJoCo_Gym.mujoco_rl import MuJoCoRL
from MuJoCo_Gym.wrappers import GymnasiumWrapper

from models import *
from wrappers import ObservationWrapper
from dynamics import *

from torch.utils.tensorboard import SummaryWriter

def make_env(config_dict, curriculum_dir_path):
    def thunk():
        env = MuJoCoRL(config_dict=config_dict)
        env = GymnasiumWrapper(env, config_dict["agents"][0])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        #env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env = ObservationWrapper(
            env,
            camera="agent/boxagent_camera",
            curriculum_directory=curriculum_dir_path,
            threshold_reward=0.5,
            make_env=make_env,
            config_dict=config_dict,
        )
        env.action_space.seed(1)
        env.observation_space.seed(1)

        return env


def test(rank, args, shared_model, config_dict, test_dir_path, checkpoint_file_path, device):
    torch.manual_seed(args.seed + rank)

    print("Starting test thread ...")
    # for debugging: pull one file from test_dir_path
    test_file = list(test_dir_path.iterdir())[0]
    test_json = test_file.with_suffix(".json")
    test_xml = test_file.with_suffix(".xml")
    config_dict["infoJson"] = str(test_json.as_posix())
    config_dict["xmlPath"] = str(test_xml.as_posix())

    # print config dict
    print("Config dict: ", config_dict)

    #env = grounding_env.GroundingEnv(args)
    env = gym.vector.AsyncVectorEnv(
        [make_env(config_dict, test_dir_path) for _ in range(1)], context="spawn", shared_memory=False
    )
    _ = env.reset() # TODO check if necessary

    model = A3C_LSTM_GA(args).to(device)

    #if (args.load != "0"):
    #    print("Loading model ... "+args.load)
    #    model.load_state_dict(
    #        torch.load(args.load, map_location=lambda storage, loc: storage))

    if args.load != "0":
        print("Loading model from ... " + str(checkpoint_file_path))
        checkpoint = torch.load(checkpoint_file_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    observation_dict, _ = env.reset()
    image = observation_dict["image"]
    instruction_idx = observation_dict["instruction_idx"]

    # Print instruction while evaluating and visualizing
    if args.evaluate != 0 and args.visualize == 1:
        print("Instruction idx: {} ".format(instruction_idx)) # TODO convert to words

    # Getting indices of the words in the instruction
    #instruction_idx = []
    #for word in instruction.split(" "):
    #    instruction_idx.append(env.word_to_idx[word])
    #instruction_idx = np.array(instruction_idx)

    #image = torch.from_numpy(image).float()/255.0
    #instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)
    image = torch.from_numpy(image).float().to(device)
    instruction_idx = torch.from_numpy(instruction_idx).to(device)


    reward_sum = 0
    done = True

    start_time = time.time()

    episode_length = 0
    rewards_list = []
    accuracy_list = []
    episode_length_list = []
    num_episode = 0
    best_reward = 0.0
    test_freq = 50
    while True:
        episode_length += 1
        if done:
            if (args.evaluate == 0):
                model.load_state_dict(shared_model.state_dict())

            cx = Variable(torch.zeros(1, 256), volatile=True).to(device)
            hx = Variable(torch.zeros(1, 256), volatile=True).to(device)
        else:
            cx = Variable(cx.data, volatile=True).to(device)
            hx = Variable(hx.data, volatile=True).to(device)

        tx = Variable(torch.from_numpy(np.array([episode_length])).long(),
                      volatile=True).to(device)

        value, logit, (hx, cx) = model(
                (Variable(image.unsqueeze(0), volatile=True),
                 Variable(instruction_idx, volatile=True), (tx, hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()

        observation, reward, truncated, terminated, _ = env.step(action[0])

        done = terminated or truncated
        done = (
            done or episode_length >= args.max_episode_length
        )
        reward_sum += reward

        if done:
            num_episode += 1
            rewards_list.append(reward_sum)
            # Print reward while evaluating and visualizing
            if args.evaluate != 0 and args.visualize == 1:
                print("Total reward: {}".format(reward_sum))

            episode_length_list.append(episode_length)
            if reward >= 1: # TODO look for max reward of all steps, instead of this!
                accuracy = 1
            else:
                accuracy = 0
            accuracy_list.append(accuracy)
            if(len(rewards_list) >= test_freq):
                print(" ".join([
                    "Time {},".format(time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time))),
                    "Avg Reward {},".format(np.mean(rewards_list)),
                    "Avg Accuracy {},".format(np.mean(accuracy_list)),
                    "Avg Ep length {},".format(np.mean(episode_length_list)),
                    "Best Reward {}".format(best_reward)]))
                logging.info(" ".join([
                    "Time {},".format(time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time))),
                    "Avg Reward {},".format(np.mean(rewards_list)),
                    "Avg Accuracy {},".format(np.mean(accuracy_list)),
                    "Avg Ep length {},".format(np.mean(episode_length_list)),
                    "Best Reward {}".format(best_reward)]))
                if np.mean(rewards_list) >= best_reward and args.evaluate == 0:
                    torch.save(model.state_dict(),
                               args.dump_location+"model_best")
                    best_reward = np.mean(rewards_list)

                rewards_list = []
                accuracy_list = []
                episode_length_list = []
            reward_sum = 0
            episode_length = 0
            reset_dict, _ = env.reset()
            image = reset_dict["image"]
            instruction_idx = reset_dict["instruction_idx"]
            # Print instruction while evaluating and visualizing
            if args.evaluate != 0 and args.visualize == 1:
                print("Instruction idx: {} ".format(instruction_idx)) # TODO convert to words

            # Getting indices of the words in the instruction
           #instruction_idx = []
           # for word in instruction.split(" "):
           #     instruction_idx.append(env.word_to_idx[word])
            #instruction_idx = np.array(instruction_idx)
            #instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)

        image = torch.from_numpy(image).float()/255.0
