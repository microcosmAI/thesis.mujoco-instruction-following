import torch.optim as optim

from models import *
from torch.autograd import Variable

import logging

from MuJoCo_Gym.mujoco_rl import MuJoCoRL
from MuJoCo_Gym.wrappers import GymnasiumWrapper, GymWrapper
from gymnasium.wrappers.frame_stack import FrameStack

import instruction_processing

from wrappers import ObservationWrapper

# from gymnasium.wrappers import NormalizeObservationV0
from dynamics import *
import argparse
import os
import random
import time
from distutils.util import strtobool
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# from wrappers.record_episode_statistics import RecordEpisodeStatistics
from progressbar import progressbar


def make_env(config_dict):
    def thunk():
        env = MuJoCoRL(config_dict=config_dict)
        env = GymnasiumWrapper(env, config_dict["agents"][0])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env = ObservationWrapper(
            env,
            camera="agent/boxagent_camera",
            curriculum_directory=os.path.join("data", "curriculum"),
            threshold_reward=0.5,
            make_env=make_env,
            config_dict=config_dict,
        )
        env.action_space.seed(1)
        env.observation_space.seed(1)

        return env

    return thunk


"""def make_base_env(config_dict):
    env = MuJoCoRL(config_dict=config_dict)
    env = GymnasiumWrapper(env, config_dict["agents"][0])
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    env.action_space.seed(1)
    env.observation_space.seed(1)

    return env


def make_env(config_dict):
    def thunk():
        env = make_base_env(config_dict)
        env = ObservationWrapper(
            env,
            camera="agent/boxagent_camera",
            curriculum_directory=os.path.join("data", "curriculum"),
            threshold_reward=0.5,
            make_env=make_base_env,
            config_dict=config_dict,
        )

        return env

    return thunk"""


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def map_discrete_to_continuous(action):
    factor = 0.1
    if action == 0:  # turn_left
        return (np.array([0]), np.array([1 * factor]))
    elif action == 1:  # turn_right
        return (np.array([0]), np.array([-1 * factor]))
    elif action == 2:  # move_forward
        return (np.array([1 * factor]), np.array([0]))
    else:
        raise ValueError("Invalid action")


def train(rank, args, shared_model, config_dict, writer):
    torch.manual_seed(args.seed + rank)
    # make env as async vector env
    env = gym.experimental.vector.AsyncVectorEnv(
        [make_env(config_dict) for _ in range(1)], context="spawn", shared_memory=False
    )

    _ = env.reset()

    model = A3C_LSTM_GA(args)

    # if args.load != "0":
    #    print(str(rank) + " Loading model ... " + args.load)
    #    model.load_state_dict(
    #        torch.load(args.load, map_location=lambda storage, loc: storage)
    #    )

    if args.load != "0":
        print(str(rank) + " Loading model ... " + args.load)
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        num_iters = checkpoint["num_iters"]
        print("Checkpoint loaded - ", num_iters, "K iters")

    model.train()

    optimizer = optim.SGD(shared_model.parameters(), lr=args.lr)

    p_losses = []
    v_losses = []

    observation_dict, _ = env.reset()
    image = observation_dict["image"]
    instruction_idx = observation_dict["instruction_idx"]

    image = torch.from_numpy(image).float()
    instruction_idx = torch.from_numpy(instruction_idx)

    done = True

    episode_length = 0
    num_iters = 0
    img_nr = 0

    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            episode_length = 0
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))

        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []
        episode_lengths = []  # NOTE added

        for step in range(args.num_steps):
            episode_length += 1
            tx = Variable(torch.from_numpy(np.array([episode_length])).long())

            value, logit, (hx, cx) = model(
                (Variable(image), Variable(instruction_idx), (tx, hx, cx))
            )
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial(
                num_samples=1
            ).data  # NOTE samples now specified due to new pytorch version
            log_prob = log_prob.gather(1, Variable(action))

            observation, reward, truncated, terminated, _ = env.step(action)
            image = observation["image"]
            image = torch.from_numpy(image).float()

            done = terminated or truncated
            done = (
                done or episode_length >= args.max_episode_length
            )  # TODO check if this is necessary

            if done:
                reset_dict, _ = env.reset()
                image = reset_dict["image"]
                instruction_idx = reset_dict["instruction_idx"]
                
                image = torch.from_numpy(image).float()
                instruction_idx = torch.from_numpy(instruction_idx)

                episode_lengths.append(episode_length)

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            tx = Variable(torch.from_numpy(np.array([episode_length])).long())

            value, _, _ = model(
                (Variable(image), Variable(instruction_idx), (tx, hx, cx))
            )
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)

        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = (
                torch.tensor(rewards[i])
                + args.gamma * values[i + 1].data
                - values[i].data
            )
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = (
                policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]
            )

        optimizer.zero_grad()

        p_losses.append(policy_loss.data[0, 0])
        v_losses.append(value_loss.data[0, 0])

        if len(p_losses) > 1000:
            num_iters += 1
            print(
                " ".join(
                    [
                        "Training thread: {}".format(rank),
                        "Num iters: {}K".format(num_iters),
                        "Avg policy loss: {}".format(np.mean(p_losses)),
                        "Avg value loss: {}".format(np.mean(v_losses)),
                    ]
                )
            )
            logging.info(
                " ".join(
                    [
                        "Training thread: {}".format(rank),
                        "Num iters: {}K".format(num_iters),
                        "Avg policy loss: {}".format(np.mean(p_losses)),
                        "Avg value loss: {}".format(np.mean(v_losses)),
                    ]
                )
            )
            p_losses = []
            v_losses = []

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        log_interval = 100
        if len(p_losses) % log_interval == 0:
            # Save model and optimizer state
            # torch.save(
            #    {
            #        "model_state_dict": model.state_dict(),
            #        "optimizer_state_dict": optimizer.state_dict(),
            #        "num_iters": num_iters,
            #    },
            #    "checkpoint.pth",
            # )
            print("Checkpoint saved - ", len(p_losses) / 1000, "K steps")

            total_steps = len(p_losses) + num_iters*1000

            median_reward = np.median(rewards[-log_interval:])
            median_episode_length = np.median(episode_lengths[-log_interval:])

            writer.add_scalar("Median Reward", median_reward, total_steps)
            writer.add_scalar(
                "Median Episode Length", median_episode_length, total_steps
            )
            writer.flush()


def train_curriculum(curriculum_dir_path, rank, args, shared_model, config_dict):
    # TODO pull from the curriculum all relevant information
    threshold_reward = 0.5  # TODO set this to a reasonable value
    level_dir_paths = sorted(
        [
            os.path.join(curriculum_dir_path, d)
            for d in os.listdir(curriculum_dir_path)
            if os.path.isdir(os.path.join(curriculum_dir_path, d))
        ]
    )
    current_reward = 0
    current_level = 0

    # curriculum loop # TODO actually make it increase levels
    #for current_level in progressbar(range(len(level_dir_paths))):
    #    while current_reward < threshold_reward:
            # pull all xml files from the current level directory
    current_level_dir_path = level_dir_paths[current_level]

    current_file_paths = [
        os.path.join(curriculum_dir_path, current_level_dir_path, file)
        for file in os.listdir(current_level_dir_path)
        if file.endswith(".xml")
    ]

    print(
        "Training on level ", current_level, "with files:", current_file_paths
    )  # debugging

    # Generate a new environment
    config_dict["xmlPath"] = current_file_paths

    # TODO return current_reward from train
    writer = SummaryWriter()
    train(rank, args, shared_model, config_dict, writer)
    writer.close()

    pass


def get_random_file(directory):
    files = os.listdir(directory)
    files = [f for f in files if f.endswith(".xml")]
    return random.choice(files)


def convert_filename_to_instruction(filename):
    return filename.split(".")[0].replace("_", " ")
