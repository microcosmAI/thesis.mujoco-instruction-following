import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import os
import random
import instruction_processing as ip


class ObservationWrapper(gym.Wrapper):
    def __init__(self, env, camera, curriculum_directory, threshold_reward):
        super().__init__(env)
        self.camera = camera
        self.current_level = 0
        self.threshold_reward = threshold_reward  # set your threshold reward here
        self.curriculum_directory = curriculum_directory
        self.max_instr_length = ip.get_max_instruction_length_from_curriculum_dir(
            self.curriculum_directory
        )
        self.word_to_idx = ip.get_word_to_idx_from_curriculum_dir(
            self.curriculum_directory
        )
        self.level_directories = sorted(
            [
                os.path.join(self.curriculum_directory, d)
                for d in os.listdir(self.curriculum_directory)
                if os.path.isdir(os.path.join(self.curriculum_directory, d))
            ]
        )

        # update observation space to match image size
        image = self.get_image(env, camera)#.flatten()

        #self.observation_space = gym.spaces.Tuple(
        #    (
        #        gym.spaces.Box(
        #            low=0, high=255, shape=image.shape, dtype=np.float32
        #        ),  # image
        #        gym.spaces.Box(
        #            low=0, high=255, shape=(1,), dtype=np.uint8
        #        ),  # instruction_idx
        #        # gym.spaces.Dict({"agent/Reward": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)})  # env_observation
        #    )
        #)

        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=image.shape, dtype=np.float32),
            'instruction_idx': gym.spaces.Box(low=0, high=255, shape=(1, self.max_instr_length), dtype=np.int64),
            # 'env_observation': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        })

        print("idx space", self.observation_space['instruction_idx']) 
        print("image space", self.observation_space['image'])  

    def get_random_file(self, directory):
        files = os.listdir(directory)
        files = [f for f in files if f.endswith(".xml")]
        return random.choice(files)

    def set_current_level(self, level):
        self.current_level = level

    def set_current_file(self, file):
        self.current_file = file
        self.current_instruction = self.convert_filename_to_instruction(file)
        self.current_instruction_idx = ip.get_instruction_idx(
            self.current_instruction, self.word_to_idx, self.max_instr_length
        )

    def set_threshold_reward(self, reward):
        self.threshold_reward = reward

    def convert_filename_to_instruction(self, filename):
        return filename.split(".")[0].replace("_", " ")

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

        """
    def step(self, action):
        _, reward, truncated, terminated, info = self.env.step(
            action
        )  # TODO check if this is correct
        image = self.get_image(self.env, self.camera)
        observation = (image, self.current_instruction_idx)
        return observation, reward, truncated, terminated, info

    def reset(self):
        # TODO build complete logic into this function
        image = self.get_image(self.env, self.camera)
        env_observation = self.env.reset()
        level_directory = self.level_directories[self.current_level]
        file = self.get_random_file(level_directory)
        self.set_current_file(file)
        # TODO set new stage
        observation = (
            image,
            self.current_instruction_idx,
        )
        return observation, env_observation[1]"""

    def step(self, action):
        _, reward, truncated, terminated, info = self.env.step(action)
        image = self.get_image(self.env, self.camera)
        observation = {
            'image': image,
            'instruction_idx': self.current_instruction_idx
        }
        return observation, reward, truncated, terminated, info

    def reset(self):
        image = self.get_image(self.env, self.camera).numpy()
        env_observation = self.env.reset()
        info = env_observation[1]
        level_directory = self.level_directories[self.current_level]
        file = self.get_random_file(level_directory)
        self.set_current_file(file)

        # print the shape of the image:
        print("image shape", image.shape)

        # dummy image to match the shape of instruction_idx
        image_cut = torch.randint(low=0, high=256, size=(1,), dtype=torch.uint8)

        observation = {
            'image': image,
            'instruction_idx': self.current_instruction_idx,
        }

        # check if the observation types and shape is the same as in the observation space
        # print the shape of instruction_idx
        print("instruction idx", self.current_instruction_idx.shape)        

        #observation = {
        #    # debugging: make equal
        #    'image': image.flatten(),
        #    'instruction_idx': self.current_instruction_idx.flatten().numpy(),
        #}

        print(" ---    reset    --- ")
        print("image shape", image.shape)
        print("instruction idx", self.current_instruction_idx.shape)
        print("instruction_idx type", type(self.current_instruction_idx))
        # print whether instruction idx is uint8 or int64
        print("instruction_idx dtype", self.current_instruction_idx.dtype)
        print("env_observation", env_observation)
        print("observation:", observation)
        print("info:", info)
        print(" --- end of reset --- ", "\n")
        #return observation, env_observation[1]
        return observation, info