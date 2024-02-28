import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import os
import random
import instruction_processing as ip
import cv2
import matplotlib as plt


class ObservationWrapper(gym.Wrapper):
    def __init__(
        self, env, camera, curriculum_directory, threshold_reward, make_env, config_dict
    ):
        super().__init__(env)
        self.camera = camera
        self.image_step = 0
        self.current_level = 0
        self.threshold_reward = threshold_reward  # TODO set this to a reasonable value
        self.curriculum_directory = curriculum_directory
        self.config_dict = config_dict
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
        image = self.get_image(env, camera)

        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=image.shape, dtype=np.float32
                ),
                "instruction_idx": gym.spaces.Box(
                    low=0, high=255, shape=(1, self.max_instr_length), dtype=np.int64
                ),
            }
        )

        self.make_env = make_env

    #def get_random_file(self, directory):  # TODO obsolete
    #    files = os.listdir(directory)
    #    files = [f for f in files if f.endswith(".xml")]
    #    return random.choice(files)

    def convert_filename_to_instruction(self, filename): 
        filename = filename.split("/")[-1]
        return filename.split(".")[0].replace("_", " ")

    def get_image(self, env, camera):
        image = env.unwrapped.environment.get_camera_data(camera)
        

        # Crop to 168x300
        height, width = image.shape[:2]
        new_height, new_width = 168, 300

        start_row = int((height - new_height) / 2)
        start_col = int((width - new_width) / 2)
        end_row = start_row + new_height
        end_col = start_col + new_width

        image = image[start_row:end_row, start_col:end_col]

        self.write_image(image)

        # TODO write image
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)  # add batch dimension

        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(0, 3, 1, 2)  # reorder for pytorch
        #image = F.interpolate(image, size=(168, 300))  # resize for the model
        # batch dimension gets added back from vector env wrapper - comment out this line if not using vector env wrapper

        image = image.squeeze(0)
        return image

    def set_instruction_idx(self, env):
        instruction = self.convert_filename_to_instruction(
            env.unwrapped.environment.xml_path
        )

        instruction_idx = []
        for word in instruction.split(" "):
            instruction_idx.append(self.word_to_idx[word])

        # Pad the instruction to the maximum instruction length using 0 as special token
        pad_length = self.max_instr_length - len(instruction_idx)
        if pad_length > 0:
            instruction_idx += [0] * pad_length
        instruction_idx = np.array(instruction_idx)
        instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)
        self.current_instruction_idx = instruction_idx


    def write_image(self, image):   
        if not os.path.exists(os.path.join(os.getcwd(), "data", "images")):
            os.makedirs(os.path.join(os.getcwd(), "data", "images"))
        
        self.image_step += 1
        if self.image_step % 1000 == 0:
            image_path = os.path.join(
                os.getcwd(), "data", "images", f"{self.image_step}.png"
            )

            cv2.imwrite(image_path, image)
            print(f"Saved image {self.image_step} to {image_path}")

    def step(self, action):
        _, reward, truncated, terminated, info = self.env.step(action)
        image = self.get_image(self.env, self.camera)

        # TODO check if I need to set instruction idx here
        observation = {"image": image, "instruction_idx": self.current_instruction_idx}
        return observation, reward, truncated, terminated, info

    def reset(self):
        self.set_instruction_idx(self.env)
        image = self.get_image(self.env, self.camera).numpy()
        env_observation = self.env.reset()
        info = env_observation[1]

        observation = {
            "image": image,
            "instruction_idx": self.current_instruction_idx,
        }

        return observation, info
