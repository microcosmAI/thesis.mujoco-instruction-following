import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import os
import random


class ObservationWrapper(gym.Wrapper):
    def __init__(self, env, camera):
        super().__init__(env)
        self.camera = camera
        self.current_level = 0
        self.threshold_reward = 100  # set your threshold reward here
        self.level_directories = sorted(
            [d for d in os.listdir(".") if os.path.isdir(d)]
        )

        # update observation space to match image size
        image = self.get_image(env, camera)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=image.shape, dtype=np.uint8
        )

    def get_random_file(self, directory):
        files = os.listdir(directory)
        return random.choice(files)

    def convert_filename_to_instruction(self, filename):
        # your function here
        pass

    def set_current_level(self, level):
        self.current_level = level

    def set_current_file(self, file):
        self.current_file = file

    def set_threshold_reward(self, reward):
        self.threshold_reward = reward

    def get_image(self, env, camera):
        image = env.unwrapped.environment.get_camera_data(camera)
        # TODO write image
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)  # add batch dimension
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(0, 3, 1, 2)  # reorder for pytorch
        image = F.interpolate(image, size=(168, 300))  # resize for the model
        # batch dimension gets added back from vector env wrapper - comment out this line if not using vector env wrapper
        image = image.squeeze(0)

        return image

    def step(self, action):
        _, reward, truncated, terminated, info = self.env.step(
            action
        )  # TODO check if this is correct
        image = self.get_image(self.env, self.camera)
        instruction = self.convert_filename_to_instruction(self.current_file)
        observation = (image, instruction)
        return observation, reward, truncated, terminated, info

    def reset(self):
        image = self.get_image(self.env, self.camera)
        env_observation = list(self.env.reset())
        level_directory = self.level_directories[self.current_level]
        file = self.get_random_file(level_directory)
        self.set_current_file(file)
        instruction = self.convert_filename_to_instruction(file)
        observation = [(image, instruction)]
        observation.extend(
            env_observation[1:]
        )  # Add the rest of the environment observation
        return tuple(observation)