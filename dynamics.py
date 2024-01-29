from MuJoCo_Gym.mujoco_rl import MuJoCoRL
import numpy as np
import cv2
import copy
from sklearn.metrics import mean_squared_error
from autoencoder import Autoencoder
from ray.air import session


class Image:
    # TODO pass through the image without autoencoder for now, later compare with autoencoder
    # TODO rewrite autoencoder in torch, not tf
    def __init__(self, environment):
        self.environment = environment
        self.observation_space = {
            "low": [0 for _ in range(50)],
            "high": [1 for _ in range(50)],
        }
        self.action_space = {"low": [], "high": []}
        self.autoencoder = Autoencoder(latent_dim=50, input_shape=(64, 64, 3))
        self.autoencoder.encoder.load_weights("models/encoder50.h5")
        self.index = 0

    def dynamic(self, agent, actions):
        self.index = self.index + 1
        image = self.environment.get_camera_data(agent + "boxagent_camera")
        image = cv2.resize(image, (64, 64))
        result = self.autoencoder.encoder.predict(np.array([image]), verbose=0)[0]
        # cv2.imwrite("/Users/cowolff/Documents/GitHub/s.mujoco_environment/ant-images/" + str(self.index) + ".png", image)
        return 0, result, 0, 0


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

        print(self.environment.agents)

        for key in self.environment.data_store.keys():
            print(self.environment.data_store[key])
        print(self.environment.filter_by_tag("Target"))
        
        if not "targets" in self.environment.data_store.keys():
            self.environment.data_store["targets"] = self.environment.filter_by_tag(
                "Target"
            )

        if not "agent" in self.environment.data_store.keys():
            self.environment.data_store["agent"] = self.environment.filter_by_tag(
                "Agent"
            )

        if not "last_distance" in self.environment.data_store.keys():
            self.environment.data_store["last_distance"] = min(
                [
                    self.environment.distance(
                        self.environment.data_store["agent"], target
                    )
                    for target in self.environment.data_store["targets"]
                ]
            )

        reward = 0
        agent = self.environment.data_store["agent"]
        targets = self.environment.data_store["targets"]

        # debugging:print agent and target
        print("agent: ", agent)
        print("targets: ", targets)

        # xml path debugging
        print("xml path: ", self.environment.xml_path)
        # json path debugging
        print("json file: ", self.environment.info_json)

        new_distance = min(
            [
                self.environment.distance(agent, target)
                for target in targets
            ]
        )
        reward = self.environment.data_store["last_distance"] - new_distance
        self.environment.data_store["last_distance"] = copy.deepcopy(new_distance)

        return reward, []

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
    # target = mujoco_gym.data_store["target"]
    reward = 0

    if mujoco_gym.collision("agent/boxagent_geom", "target"):
        return 1

    return reward


"""def collision_reward(mujoco_gym, agent):
    for border in ["border_geom", "border1_geom", "border2_geom", "border3_geom"]:
        
        if mujoco_gym.collision(border, "agent/boxagent_geom"):
            return -0.1
        
    return 0"""


def target_done(mujoco_gym, agent):
    if mujoco_gym.collision("target", "agent/boxagent_geom"):
        return True
    return False


"""def border_done(mujoco_gym, agent):
    for border in ["border_geom", "border1_geom", "border2_geom", "border3_geom"]:
        if mujoco_gym.collision(border, agent + "_geom"):
            return True
    return False"""
