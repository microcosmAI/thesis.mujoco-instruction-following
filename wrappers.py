import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F


class ImageWrapper(gym.Wrapper):
    def __init__(self, env, camera):
        super().__init__(env)
        self.camera = camera
        # update observation space to match image size
        image = self.get_image(env, camera)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=image.shape, dtype=np.uint8
        )

    def step(self, action):
        # TODO check if this is the right way to get the observation or if done is not truncated, terminated
        _, reward, truncated, terminated, info = self.env.step(
            action
        )  # TODO check if this is correct
        observation = self.get_image(self.env, self.camera)
        return observation, reward, truncated, terminated, info

    def reset(self):
        image = self.get_image(self.env, self.camera)
        env_observation = list(self.env.reset())
        observation = [image]
        observation.extend(env_observation[1:])
        return tuple(observation)

    """def get_image(self, env, camera):
        image = env.env.environment.get_camera_data(camera)
        if len(image.shape) == 3:
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1)  # reorder for pytorch
            image = F.interpolate(
                image.unsqueeze(0), size=(168, 300)
            )  # resize for the model 
            image = image.squeeze(0)  # remove batch dimension before returning (gets added back from vector env wrapper)
        # Debugging
        print("Image exported from get_image with shape: ", image.shape)

        return image"""

    def get_image(self, env, camera):
        image = env.env.environment.get_camera_data(camera)
        # TODO write image
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)  # add batch dimension
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(0, 3, 1, 2)  # reorder for pytorch
        image = F.interpolate(image, size=(168, 300))  # resize for the model
        image = image.squeeze(
            0
        )  # remove batch dimension before returning (gets added back from vector env wrapper - comment out this line if not using vector env wrapper)

        return image
