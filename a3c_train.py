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
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 20))
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

    return thunk


def ensure_shared_grads(model, shared_model, device):
    """
    Ensures that the gradients of the shared model are synchronized with the gradients of the local model

    Args:
        model (torch.nn.Module): The local model whose gradients need to be synchronized
        shared_model (torch.nn.Module): The shared model whose gradients will be updated
        device (torch.device): The device on which the gradients will be moved
    """
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            shared_param.grad = shared_param.grad.to(device)
            return
        shared_param._grad = param.grad
        if shared_param._grad is not None:
            shared_param._grad = shared_param._grad.to(device)


def train(
    rank,
    args,
    shared_model,
    config_dict,
    writer,
    curriculum_dir_path,
    checkpoint_file_path,
    device,
    threshold_accuracy=0.9,
):
    torch.manual_seed(args.seed + rank)

    # make env as async vector env
    env = gym.vector.AsyncVectorEnv(
        [make_env(config_dict, curriculum_dir_path) for _ in range(1)],
        context="spawn",
        shared_memory=True,  # TODO true
    )

    _ = env.reset()

    model = A3C_LSTM_GA(args, device).to(device)  # TODO pass device into function

    # if args.load != "0":
    #    print(str(rank) + " Loading model ... " + args.load)
    #    model.load_state_dict(
    #        torch.load(args.load, map_location=lambda storage, loc: storage)
    #    )

    if args.load != "0":  # TODO fix this
        print(str(rank) + " Loading model ... " + args.load)
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        num_iters = checkpoint["num_iters"]
        pass

    model.train()

    optimizer = optim.SGD(shared_model.parameters(), lr=args.lr)

    p_losses = []
    v_losses = []

    observation_dict, _ = env.reset()
    image = observation_dict["image"]
    instruction_idx = observation_dict["instruction_idx"]

    image = torch.from_numpy(image).float().to(device)
    instruction_idx = torch.from_numpy(instruction_idx).to(device)

    done = True

    episode_length = 0
    episode_lengths = []  # NOTE added
    accuracies = []
    avg_accuracy = 0  # sliding window
    avg_accuracy_window_size = 10  # nr of episodes
    threshold_reached = False
    num_iters = 0
    first_iter = True

    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        if done:

            if not first_iter:
                episode_lengths.append(episode_length)
            first_iter = False

            episode_length = 0
            cx = Variable(torch.zeros(1, 256)).to(device)
            hx = Variable(torch.zeros(1, 256)).to(device)

        else:
            cx = Variable(cx.data).to(device)
            hx = Variable(hx.data).to(device)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1
            tx = Variable(torch.from_numpy(np.array([episode_length])).long()).to(
                device
            )

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
            image = torch.from_numpy(image).float().to(device)

            done = terminated or truncated
            done = (
                done or episode_length >= args.max_episode_length
            )  # TODO check if this is necessary

            if done:
                reset_dict, _ = env.reset()
                image = reset_dict["image"]
                instruction_idx = reset_dict["instruction_idx"]

                image = torch.from_numpy(image).float().to(device)
                instruction_idx = torch.from_numpy(instruction_idx).to(device)

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                accuracies.append(track_metrics(
                    writer, p_losses, rewards, episode_lengths
                ))
                if len(accuracies) > avg_accuracy_window_size:
                    avg_accuracy = (
                        sum(accuracies[-avg_accuracy_window_size:])
                        / avg_accuracy_window_size
                    )
                    print("Average accuracy: ", avg_accuracy)
                    logging.info("Average accuracy: " + str(avg_accuracy))

                if avg_accuracy > threshold_accuracy:
                    threshold_reached = True
                    print("Threshold accuracy reached")
                    break

                break

        R = torch.zeros(1, 1)
        if not done:
            tx = Variable(torch.from_numpy(np.array([episode_length])).long()).to(
                device
            )

            value, _, _ = model(
                (Variable(image), Variable(instruction_idx), (tx, hx, cx))
            )
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R).to(device)

        gae = torch.zeros(1, 1).to(device)
        for i in reversed(range(len(rewards))):
            reward = torch.tensor(rewards[i]).to(
                device
            )  # TODO check if np.array(rewards) is faster
            values = [
                value.item() if torch.is_tensor(value) else value for value in values
            ]
            values = torch.tensor(values).to(device)
            R = args.gamma * R + reward
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = reward + args.gamma * values[i + 1].data - values[i].data

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

            # checkpoint: save the model for rank 0
            if rank == 0:
                torch.save(
                    {
                        "model_state_dict": shared_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "num_iters": num_iters,
                    },
                    checkpoint_file_path,
                )

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        ensure_shared_grads(model, shared_model, device)
        optimizer.step()

        if threshold_reached:
            break


def train_curriculum(
    curriculum_dir_path,
    rank,
    args,
    shared_model,
    config_dict,
    checkpoint_file_path,
    device,
):
    # TODO pull from the curriculum all relevant information
    threshold_accuracy = 0.8  # TODO set this to a reasonable value
    curriculum_dir_path = Path(curriculum_dir_path)
    level_dir_paths = sorted(
        [
            str(curriculum_dir_path / d)
            for d in curriculum_dir_path.iterdir()
            if (curriculum_dir_path / d).is_dir()
        ]
    )
    current_reward = 0
    current_level = 0

    writer = SummaryWriter()

    # curriculum loop # TODO actually make it increase levels
    for current_level in range(len(level_dir_paths)):

        if current_level != 0:
            args.load = "1"

        # pull all xml files from the current level directory
        current_level_dir_path = Path(level_dir_paths[current_level])

        xml_files = [
            file for file in current_level_dir_path.iterdir() if file.suffix == ".xml"
        ]
        json_files = [
            file.with_suffix(".json") for file in xml_files
        ]  # NOTE not filtered from dir, to exclude prompts.json
        config_dict["xmlPath"] = [str(file.as_posix()) for file in xml_files]
        config_dict["infoJson"] = [str(file.as_posix()) for file in json_files]

        print(
            "Training on level ",
            current_level,
            "with files:",
            xml_files,
            " / ",
            config_dict["infoJson"],
        )  # debugging

        train(
            rank,
            args,
            shared_model,
            config_dict,
            writer,
            curriculum_dir_path,
            checkpoint_file_path,
            device,
            threshold_accuracy,
        )
        writer.close()

        pass

    writer.close()


def track_metrics(writer, p_losses, rewards, episode_lengths):

    if episode_lengths:
        total_steps = sum(episode_lengths)
        print(total_steps / 1000, "K steps")

        total_reward = sum(rewards)
        avg_reward = total_reward / len(rewards)
        median_reward = np.median(rewards)
        max_reward = max(rewards)
        min_reward = min(rewards)

        if max_reward > 1:
            accuracy = 1
        else:
            accuracy = 0

        if len(p_losses) == 0:
            p_losses.append(0)
        avg_p_loss = sum(p_losses) / len(p_losses)
        max_p_loss = max(p_losses)
        min_p_loss = min(p_losses)

        min_episode_length = min(episode_lengths)

        last_episode_length = episode_lengths[-1]

        writer.add_scalar("Total Reward", total_reward, total_steps)
        writer.add_scalar("Average Reward", avg_reward, total_steps)
        writer.add_scalar("Median Reward", median_reward, total_steps)
        writer.add_scalar("Max Reward", max_reward, total_steps)
        writer.add_scalar("Min Reward", min_reward, total_steps)

        writer.add_scalar("Average Policy Loss", avg_p_loss, total_steps)
        writer.add_scalar("Max Policy Loss", max_p_loss, total_steps)
        writer.add_scalar("Min Policy Loss", min_p_loss, total_steps)

        writer.add_scalar("Episode Length", last_episode_length, total_steps)
        writer.add_scalar("Min Episode Length", min_episode_length, total_steps)

        writer.flush()

        return accuracy
    
    else:
        return 0
