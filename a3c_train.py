import torch.optim as optim

from models import *
from torch.autograd import Variable

import logging

from MuJoCo_Gym.mujoco_rl import MuJoCoRL
from MuJoCo_Gym.wrappers import GymnasiumWrapper, GymWrapper
from gymnasium.wrappers.frame_stack import FrameStack

import instruction_processing

from wrappers import ImageWrapper

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
        env = ImageWrapper(env, camera="agent/boxagent_camera")
        env.action_space.seed(1)
        env.observation_space.seed(1)

        return env

    return thunk


def make_only_env(config_dict):
    def thunk():
        env = MuJoCoRL(config_dict=config_dict)

        return env

    return thunk


def wrap_env(env, config_dict):
    env = GymnasiumWrapper(env, config_dict["agents"][0])
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    env.action_space.seed(1)
    env.observation_space.seed(1)
    # wrap in syncvectorenv # NOTE must use syncvectorenv to access individual env functions
    env = gym.vector.AsyncVectorEnv([lambda: make_only_env(config_dict)()], context="spawn")
    #env = gym.vector.SyncVectorEnv([lambda: make_only_env(config_dict)()])
    return env


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def get_image(env, camera):
    image = env.env.environment.get_camera_data(camera) # TODO env.env or env.environment
    # TODO write image
    image = np.expand_dims(image, 0)  # add batch size dimension
    image = torch.from_numpy(image).float() / 255.0
    image = image.permute(0, 3, 1, 2)  # reorder for pytorch
    image = F.interpolate(image, size=(168, 300)) # resize for the model TODO look into getting different camera resolutions
    return image


def get_instruction_idx(instruction, word_to_idx):
    instruction_idx = []
    for word in instruction.split(" "):
        instruction_idx.append(word_to_idx[word])
    instruction_idx = np.array(instruction_idx)
    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)
    return instruction_idx

def map_discrete_to_continuous(action):
    if action == 0:  # turn_left
        return np.array([0, 0, 0, 0, 1, 0])
    elif action == 1:  # turn_right
        return np.array([0, 0, 0, 0, -1, 0])
    elif action == 2:  # move_forward
        return np.array([1, 0, 0, 0, 0, 0])
    else:
        raise ValueError("Invalid action")


def train(rank, args, shared_model, config_dict):
    torch.manual_seed(args.seed + rank)

    #env = make_only_env(config_dict)()

    #env = wrap_env(env, config_dict)

    # make env as async vector env
    env = gym.vector.AsyncVectorEnv([make_env(config_dict) for _ in range(1)], context="spawn")

    # get word_to_idx # TODO rename word_to_idx because it is a dumb name
    word_to_idx = instruction_processing.get_word_to_idx_from_dir(  
        os.path.join(os.getcwd(), "xml_debug_files")
    )

    # debugging: print reset returns
    print(env.reset())
    env.reset()

    model = A3C_LSTM_GA(args)

    if args.load != "0":
        print(str(rank) + " Loading model ... " + args.load)
        model.load_state_dict(
            torch.load(args.load, map_location=lambda storage, loc: storage)
        )

    model.train()

    optimizer = optim.SGD(shared_model.parameters(), lr=args.lr)

    p_losses = []
    v_losses = []

    # TODO here is where the a3c implementation gets the image and instruction from the environment
    observation = env.reset()
    print("Observation: ")
    print(observation)
    # The instruction is the infoJsons file name # TODO this might change later
    # TODO figure out a way to get current instruction with each reset
    instruction = config_dict["infoJson"].split("/")[-1].split(".")[0].replace("_", " ")
    image = get_image(env=env, camera="agent/boxagent_camera")

    instruction_idx = get_instruction_idx(instruction, word_to_idx)

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

        for step in range(args.num_steps):
            if step % 100 == 0:
                # write image to disk
                img = get_image(env=env, camera="agent/boxagent_camera")
                # store image in folder as image_{step}.png
                img = img.permute(0, 2, 3, 1).numpy()
                img = np.squeeze(img)
                img = (img * 255).astype(np.uint8)
                name = "image_{}.png".format(img_nr)
                # write to ./images dir
                cv2.imwrite(os.path.join(os.getcwd(), "images", name), img)
                img_nr += 1

            episode_length += 1
            tx = Variable(torch.from_numpy(np.array([episode_length])).long())

            value, logit, (hx, cx) = model(
                (Variable(image), Variable(instruction_idx), (tx, hx, cx))
            )
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).data # NOTE samples now specified due to new pytorch version
            log_prob = log_prob.gather(1, Variable(action))


            image = get_image(env=env, camera="agent/boxagent_camera")

            print(env.step(action))
            _, reward, termination, truncation, _ = env.step(action) # TODO check if this is correct

            done = termination or truncation
            done = done or episode_length >= args.max_episode_length # TODO check if this is necessary

            if done:
                env.reset()
                #(image, instruction), _, _, _ = env.reset()
                image = get_image(env=env, camera="agent/boxagent_camera")
                # TODO: get the current instruction
                instruction = config_dict["infoJson"].split("/")[-1].split(".")[0].replace("_", " ")
                instruction_idx = get_instruction_idx(instruction, word_to_idx)

            image = get_image(env=env, camera="agent/boxagent_camera")

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

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
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
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()


def main():
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
    }

    envs = gym.vector.AsyncVectorEnv(
        [make_env(config_dict) for _ in range(num_envs)], context="spawn"
    )
    # print type of action space
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # print action and observation space details and shapes
    print("Action space: ", envs.single_action_space)
    print("Observation space: ", envs.single_observation_space)
    print("Action space shape: ", envs.single_action_space.shape)
    print("Observation space shape: ", envs.single_observation_space.shape)


    # set up logging
    logging.basicConfig(
        filename="a3c.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # set up tensorboard
    writer = SummaryWriter()

    # set up model
    shared_model = A3C_LSTM_GA(envs.single_observation_space, envs.single_action_space)
    shared_model.share_memory()

    # set up optimizer
    optimizer = optim.Adam(shared_model.parameters(), lr=1e-4)

    # set up training
    processes = []
    for rank in range(0, 1):
        p = torch.multiprocessing.Process(
            target=train, args=(rank, args, shared_model, config_dict)
        )
        p.start()
        processes.append(p)


if __name__ == "__main__":
    main()
