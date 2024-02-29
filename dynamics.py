# from MuJoCo_Gym.mujoco_rl import MuJoCoRL
# import numpy as np
# import cv2
import copy

# from sklearn.metrics import mean_squared_error
# from autoencoder import Autoencoder
import re

# import os

"""class Image:
    # TODO pass through the image without autoencoder for now, later compare with autoencoder
    # TODO rewrite autoencoder in torch, not tf
    def __init__(self, environment):
        self.environment = environment
        #self.observation_space = {"low": [0 for _ in range(50)], "high": [1 for _ in range(50)]}
        # make the observation space a box of 64x64x3
        self.observation_space = {"low": [0 for _ in range(256*256*3)], "high": [1 for _ in range(256*256*3)]}
        self.action_space = {"low": [], "high": []}
        #self.autoencoder = Autoencoder(latent_dim=50, input_shape=(256*256*3))
        #self.autoencoder.encoder.load_weights("models/encoder50.h5")
        self.index = 0
      

    def dynamic(self, agent, actions):
        self.index = self.index + 1
        image = self.environment.get_camera_data(agent + "boxagent_camera")
        image = cv2.resize(image, (256, 256))
        #result = self.autoencoder.encoder.predict(np.array([image]), verbose=0)[0]
        result = image.flatten() # TODO remove this line, inspect result
        # set filepath to current filepath + "debug images"
        filepath = os.path.join(os.getcwd(), "debug_images")
        cv2.imwrite(filepath + str(self.index) + ".png", image)
        return 0, result, 0, 0"""


class Reward:
    def __init__(self, environment):
        self.environment = environment
        self.observation_space = {"low": [], "high": []}
        self.action_space = {"low": [], "high": []}
        self.environment.data_store["last_distance"] = 0

    """def dynamic(self, agent, actions):
        
        # NOTE: only built for envs with a single target in them
        if "target" not in self.environment.data_store[agent].keys():
            self.environment.data_store["target"] = self.environment.filter_by_tag("target")[0]

        reward = 0

        target = self.environment.data_store["target"]
        new_distance = self.environment.distance("agent/boxagent_geom", "target")

        # TODO: Beware, here there be duct tape and spit
        try:
            reward = self.environment.data_store["last_distance"] - new_distance
        except KeyError:
            self.environment.data_store["last_distance"] = 0
            reward = 0
            print(" --- tried so far, got so hard --- ")

        
        self.environment.data_store["last_distance"] = copy.deepcopy(new_distance)
        return reward, []"""

    def dynamic(self, agent, actions):
        if not "targets" in self.environment.data_store.keys():
            self.environment.data_store["targets"] = self.environment.filter_by_tag(
                "Target"
            )

        if not "target_geoms" in self.environment.data_store.keys():
            self.environment.data_store["target_geoms"] = []

            for target in self.environment.data_store["targets"]:
                # TODO this is a hack, necessary because of the way the targets are named in the json/xml files

                suffix = re.split("\d", target["name"])[0] + "_geom"
                prefix = re.split("/", target["name"])[0]

                target_geom_name = prefix + "/" + suffix

                self.environment.data_store["target_geoms"].append(target_geom_name)

        if not "agent" in self.environment.data_store.keys():
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

    # def dynamic(self, agent, actions):
    #    # Minimal version that just returns the same reward for now
    #    print("i'm  t - n -t , i'm  dy - na - mic")
    #    if "target" not in self.environment.data_store[agent].keys():
    #        self.environment.data_store["target"] = self.environment.filter_by_tag("target")[0]
    #
    #    return 1, []


"""def turn_done(mujoco_gym, agent):
    _healthy_z_range = (0.35, 1.1)
    if mujoco_gym.data.body(agent).xipos[2] < _healthy_z_range[0] or mujoco_gym.data.body(agent).xipos[2] > _healthy_z_range[1]:
        return True
    else:
        return False"""

"""def turn_reward(mujoco_gym, agent):
    _healthy_z_range = (0.35, 1.1)
    if mujoco_gym.data.body(agent).xipos[2] < _healthy_z_range[0] or mujoco_gym.data.body(agent).xipos[2] > _healthy_z_range[1]:
        return -0.5
    else:
        return 0"""


def target_reward(mujoco_gym, agent):
    """1 if agent is colliding with target, 0 otherwise"""
    targets = mujoco_gym.data_store["target_geoms"]

    reward = 0

    for target in targets:
        if mujoco_gym.collision(target, agent + "boxagent_geom"):
            reward = 1
            break

    return reward


"""def collision_reward(mujoco_gym, agent):
    for border in ["border_geom", "border1_geom", "border2_geom", "border3_geom"]:
        
        if mujoco_gym.collision(border, "agent/boxagent_geom"):
            return -0.1
        
    return 0"""


def target_done(mujoco_gym, agent):
    """True if agent is colliding with target, False otherwise"""
    targets = mujoco_gym.data_store["target_geoms"]

    for target in targets:
        if mujoco_gym.collision(target, agent + "boxagent_geom"):
            return True
        else:
            return False


"""def border_done(mujoco_gym, agent):
    for border in ["border_geom", "border1_geom", "border2_geom", "border3_geom"]:
        if mujoco_gym.collision(border, agent + "_geom"):
            return True
    return False"""
