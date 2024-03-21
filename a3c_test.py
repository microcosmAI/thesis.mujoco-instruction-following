import numpy as np
import torch
import torch.nn.functional as F
import time
from models import A3C_LSTM_GA

from torch.autograd import Variable
from constants import *

from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

from MuJoCo_Gym.mujoco_rl import MuJoCoRL
from MuJoCo_Gym.wrappers import GymnasiumWrapper

from wrappers import ObservationWrapper
from dynamics import *

from torch.utils.tensorboard import SummaryWriter


def make_env(config_dict, curriculum_dir_path):
    def thunk():
        env = MuJoCoRL(config_dict=config_dict)
        env = GymnasiumWrapper(env, config_dict["agents"][0])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = ObservationWrapper(
            env,
            camera="agent/boxagent_camera",
            curriculum_directory=curriculum_dir_path, # NOTE NOT the test directory
            threshold_reward=0.5,
            make_env=make_env,
            config_dict=config_dict,
        )
        env.action_space.seed(1)
        env.observation_space.seed(1)
        print("Env created ...")
        return env

    return thunk


def test(
    rank, args, shared_model, config_dict, curriculum_dir_path, test_dir_path, checkpoint_file_path, device
):
    torch.manual_seed(args.seed + rank)

    if args.visualize:
        # pull only the first stage from the test set, because renderMode will only work with a single stage
        test_file_path = list(test_dir_path.iterdir())[0]
        print("Testing on file: ", test_file_path)
        test_json = test_file_path.with_suffix(".json")
        test_xml = test_file_path.with_suffix(".xml")
        config_dict["infoJson"] = str(test_json.as_posix())
        config_dict["xmlPath"] = str(test_xml.as_posix())
        config_dict["renderMode"] = True
        config_dict["environmentDynamics"] = [],

    else:
        # get all stages from the test set
        test_dir_path = Path(test_dir_path)
        xml_files = [file for file in test_dir_path.iterdir() if file.suffix == ".xml"]
        json_files = [file.with_suffix(".json") for file in xml_files] # NOTE not filtered from dir, to exclude prompts.json
        config_dict["xmlPath"] = [str(file.as_posix()) for file in xml_files]
        config_dict["infoJson"] = [str(file.as_posix()) for file in json_files]


    env = gym.vector.AsyncVectorEnv(
        [make_env(config_dict, curriculum_dir_path) for _ in range(1)],
        context="spawn",
        shared_memory=False,
    )

    _ = env.reset() 

    model = A3C_LSTM_GA(args, device).to(device)

    print("Loading model from: " + str(checkpoint_file_path))
    checkpoint = torch.load(
        checkpoint_file_path, map_location=lambda storage, loc: storage
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # set tensorboard writer to write to directory runs/test{current_time}
    writer = SummaryWriter("runs/test" + time.strftime("%Y-%m-%d-%H-%M-%S"))

    observation_dict, _ = env.reset()
    image = observation_dict["image"]
    image = torch.from_numpy(image).float().to(device)
    instruction_idx = observation_dict["instruction_idx"]
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
    total_steps = 0 # steps, actually

    while True:
        episode_length += 1
        total_steps += 1

        if done:
            with torch.no_grad():
                cx = torch.zeros(1, 256).to(device)
                hx = torch.zeros(1, 256).to(device)
        else:
            with torch.no_grad():
                cx = cx.data.to(device)
                hx = hx.data.to(device)

        with torch.no_grad():
            tx = torch.from_numpy(np.array([episode_length])).long().to(device)

        with torch.no_grad():
            value, logit, (hx, cx) = model((image, instruction_idx, (tx, hx, cx)))
            prob = F.softmax(logit, dim=1)
            action = prob.max(1)[1].numpy()

        observation_dict, reward, truncated, terminated, _ = env.step([action[0]])
        image = observation_dict["image"]
        image = torch.from_numpy(image).float().to(device)

        done = terminated or truncated
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        if done:
            num_episode += 1
            rewards_list.append(reward_sum)
            # Print reward while evaluating and visualizing
            if args.evaluate != 0 and args.visualize == 1:
                print("Total reward: {}".format(reward_sum))

            episode_length_list.append(episode_length)
            if reward_sum >= 1: 
                accuracy = 1
            else:
                accuracy = 0
            accuracy_list.append(accuracy)

            track_test_metrics(writer, rewards_list, episode_length_list, total_steps)

            if len(rewards_list) >= test_freq:
                print(
                    " ".join(
                        [
                            "Time {},".format(
                                time.strftime(
                                    "%Hh %Mm %Ss", time.gmtime(time.time() - start_time)
                                )
                            ),
                            "Avg Reward {},".format(np.mean(rewards_list)),
                            "Avg Accuracy {},".format(np.mean(accuracy_list)),
                            "Avg Ep length {},".format(np.mean(episode_length_list)),
                            "Best Reward {}".format(best_reward),
                        ]
                    )
                )
                if np.mean(rewards_list) >= best_reward and args.evaluate == 0:
                    torch.save(model.state_dict(), args.dump_location + "model_best")
                    best_reward = np.mean(rewards_list)

                rewards_list = []
                accuracy_list = []
                episode_length_list = []
            reward_sum = 0
            episode_length = 0
            observation_dict, _ = env.reset()
            image = observation_dict["image"]
            image = torch.from_numpy(image).float().to(device)
            instruction_idx = observation_dict["instruction_idx"]
            instruction_idx = torch.from_numpy(instruction_idx).to(device)

            # Print instruction while evaluating and visualizing
            if args.evaluate != 0 and args.visualize == 1:
                print(
                    "Instruction idx: {} ".format(instruction_idx)
                )  

                

def track_test_metrics(writer, rewards, episode_lengths, total_steps):
    """
    Track and log test metrics using a SummaryWriter.

    Args:
        writer (SummaryWriter): The SummaryWriter object used for logging.
        rewards (list): A list of rewards obtained in each episode.
        episode_lengths (list): A list of lengths of each episode.

    Returns:
        int: The accuracy value, which is 1 if the maximum reward is greater than 1, otherwise 0.
    """

    if episode_lengths:

        total_reward = sum(rewards)
        avg_reward = total_reward / len(rewards)
        median_reward = np.median(rewards)
        max_reward = max(rewards)
        min_reward = min(rewards)

        if max_reward > 1:
            accuracy = 1
        else:
            accuracy = 0

        last_episode_length = episode_lengths[-1]

        writer.add_scalar("Total Reward", total_reward, total_steps)
        writer.add_scalar("Average Reward", avg_reward, total_steps)
        writer.add_scalar("Median Reward", median_reward, total_steps)
        writer.add_scalar("Max Reward", max_reward, total_steps)
        writer.add_scalar("Min Reward", min_reward, total_steps)

        writer.add_scalar("Episode Length", last_episode_length, total_steps)

        writer.flush()

        return accuracy
    
    else:
        return 0
