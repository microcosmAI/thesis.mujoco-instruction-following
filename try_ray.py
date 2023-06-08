from MuJoCo_Gym.mujoco_rl import MuJoCo_RL
from MuJoCo_Gym.single_agent_wrapper import Single_Agent_Wrapper

import ray
from ray.rllib.algorithms.sac import SACConfig
from ray import tune
from ray.tune.logger import pretty_print

import random
import numpy as np


def reward_function(mujoco_gym, agent):
    # Creates all the necessary fields to store the needed data within the dataStore at timestep 0 
    if "targets" not in mujoco_gym.dataStore[agent].keys():
        mujoco_gym.dataStore["targets"] = mujoco_gym.filterByTag("target")
        mujoco_gym.dataStore[agent]["current_target"] = mujoco_gym.dataStore["targets"][random.randint(0, len(mujoco_gym.dataStore["targets"]) - 1)]["name"]
        distance = mujoco_gym.distance(agent, mujoco_gym.dataStore[agent]["current_target"])
        mujoco_gym.dataStore[agent]["distance"] = distance
        new_reward = 0
    else: # Calculates the distance between the agent and the current target
        distance = mujoco_gym.distance(agent, mujoco_gym.dataStore[agent]["current_target"])
        new_reward = mujoco_gym.dataStore[agent]["distance"] - distance
        mujoco_gym.dataStore[agent]["distance"] = distance
    reward = new_reward * 10
    return reward

def done_function(mujoco_gym, agent):    
    if mujoco_gym.dataStore[agent]["distance"] <= 0.1:
        return True
    else:
        return False
    
environment_path = "Environment/SingleBoxEnv.xml" # File containing the mujoco environment
info_path = "Environment/info_example.json"    # File containing addtional environment informations
agents = ["agent1_torso"]


# Path, Info and Agents need to be passed on as a dict - this is a ray thing
config_dict = {
    "xmlPath":environment_path, 
    "infoJson":info_path, 
    "agents":agents, 
    "rewardFunctions":[reward_function], 
    "doneFunctions":[done_function], 
    "renderMode":False,
    "freeJoint":True
    }


#gymEnvironment = Single_Agent_Wrapper(env, agents[0])

ray.init()

config = SACConfig()
config = config.environment(env=MuJoCo_RL, env_config=config_dict)
config = config.resources(num_gpus=0)
config = config.rollouts(num_rollout_workers=4)
print(pretty_print(config.to_dict()))
algo = config.build()

result = algo.train()
print(" --- RESULT: --- ")
print(pretty_print(result))

