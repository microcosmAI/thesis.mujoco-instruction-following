# from MuJoCo_Gym.mujoco_rl import MuJoCoRL
# import numpy as np
# import cv2
import copy
# from sklearn.metrics import mean_squared_error
# from autoencoder import Autoencoder
import re


class Reward:
    def __init__(self, environment):
        self.environment = environment
        self.observation_space = {"low": [], "high": []}
        self.action_space = {"low": [], "high": []}
        self.environment.data_store["last_distance"] = 0


    def dynamic(self, agent, actions):
        if not "targets" in self.environment.data_store.keys():
            if self.environment.filter_by_tag("Target") == []:
                raise Exception("No targets found in environment")
            self.environment.data_store["targets"] = self.environment.filter_by_tag(
                "Target"
            )

        if not "target_geoms" in self.environment.data_store.keys():
            self.environment.data_store["target_geoms"] = []

            for target in self.environment.data_store["targets"]:
                # NOTE necessary because of the way the targets are named in the json/xml files
                # watch out for this if the naming convention changes

                suffix = re.split("\d", target["name"])[0] + "_geom"
                prefix = re.split("/", target["name"])[0]

                target_geom_name = prefix + "/" + suffix

                self.environment.data_store["target_geoms"].append(target_geom_name)

        if not "agent" in self.environment.data_store.keys():

            if self.environment.filter_by_tag("Agent") == []:
                raise Exception("No agent found in environment")
            self.environment.data_store["agent"] = self.environment.filter_by_tag(
                "Agent"
            )[0]

        if not "last_distance" in self.environment.data_store.keys():
            agent = self.environment.data_store["agent"]
            targets = self.environment.data_store["targets"]

            self.environment.data_store["last_distance"] = min(
                [
                    self.environment.distance(agent["position"], target["position"])
                    for target in targets
                ]
            )

        reward = 0
        agent = self.environment.data_store["agent"]
        targets = self.environment.data_store["targets"]

        new_distance = min(
            [
                self.environment.distance(agent["position"], target["position"])
                for target in targets
            ]
        )
        reward = self.environment.data_store["last_distance"] - new_distance
        self.environment.data_store["last_distance"] = copy.deepcopy(new_distance)

        return reward, [], 0, 0


def target_reward(mujoco_gym, agent):
    """1 if agent is colliding with target, 0 otherwise"""
    targets = mujoco_gym.data_store["target_geoms"]

    reward = 0

    for target in targets:
        if mujoco_gym.collision(target, agent + "boxagent_geom"):
            print("collision with target")
            reward = 1
            break

    return reward


def collision_reward(mujoco_gym, agent):
    for border in ["border/border_geom", "border_1/border_geom", "border_2/border_geom", "border_3/border_geom"]:
        
        if mujoco_gym.collision(border, "agent/boxagent_geom"):
            print("collision with border")
            return -0.1
        
    return 0


def target_done(mujoco_gym, agent):
    """True if agent is colliding with target, False otherwise"""
    targets = mujoco_gym.data_store["target_geoms"]

    for target in targets:
        if mujoco_gym.collision(target, agent + "boxagent_geom"):
            return True
            
        else:
            return False


def border_done(mujoco_gym, agent):
    for border in ["border/border_geom", "border_1/border_geom", "border_2/border_geom", "border_3/border_geom"]:
        if mujoco_gym.collision(border, "agent/boxagent_geom"):
            return True
    return False
