import gymnasium as gym
import numpy as np
import torch
import instruction_processing as ip
import cv2
from pathlib import Path


class ObservationWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        camera,
        curriculum_directory,
        threshold_reward,
        make_env,
        config_dict,
        test_mode=False,
    ):
        super().__init__(env)
        self.camera = camera
        self.image_step = 0

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
                str(self.curriculum_directory / d)
                for d in self.curriculum_directory.iterdir()
                if (self.curriculum_directory / d).is_dir()
            ]
        )

        # update observation space to match image size
        image = self.get_image(env, camera)

        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=1, shape=image.shape, dtype=np.float32
                ),
                "instruction_idx": gym.spaces.Box(
                    low=0, high=255, shape=(1, self.max_instr_length), dtype=np.int64
                ),
            }
        )

        self.make_env = make_env

    def get_image(self, env, camera):
        """
        Retrieves an image from the environment's camera and performs preprocessing.

        Args:
            env (object): The environment object.
            camera (str): The camera name.

        Returns:
            torch.Tensor: The preprocessed image tensor.

        """
        image = env.unwrapped.environment.get_camera_data(camera)

        # Crop to 168x300
        height, width = image.shape[:2]
        new_height, new_width = 168, 300

        start_row = int((height - new_height) / 2)
        start_col = int((width - new_width) / 2)
        end_row = start_row + new_height
        end_col = start_col + new_width

        image = image[start_row:end_row, start_col:end_col]

        self.write_image(image=image, interval=100000)

        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)  # add batch dimension

        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(0, 3, 1, 2)  # reorder for pytorch

        # batch dimension gets added back from vector env wrapper - comment out this line if not using vector env wrapper
        image = image.squeeze(0)

        return image

    def write_image(self, image, interval):
        """Writes the given image to a file if the image step is a multiple of the specified interval."""
        images_dir = Path.cwd() / "data" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        self.image_step += 1
        if (self.image_step - 900) % interval == 0:
            image_path = images_dir / f"{self.image_step}.png"

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
            cv2.imwrite(str(image_path), image)
            print(f"Saved image {self.image_step} to {str(image_path)}")

    def convert_filename_to_instruction(self, filename):
        """Converts a filename to an instruction by removing path, file extension, and hyphens, and returns it"""

        filename = filename.split("/")[-1]
        return filename.split(".")[0].replace("-", " ")

    def set_instruction_idx(self, env):
        """Sets the instruction index for the given environment, handles padding and conversion to tensor"""

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

    def map_discrete_to_continuous(self, action):
        """Maps a discrete action to a continuous action vector."""
        factor = 1.1
        rot_factor = 1.0  # factor for rotation actions (3rd value in action vector)

        # for 6 discrete actions
        if action == 0:  # forward
            return np.array([1.0 * factor, 0.0, 0.0], dtype=np.float32)
        elif action == 1:  # backward
            return np.array([-1.0 * factor, 0.0, 0.0], dtype=np.float32)
        elif action == 4:  # sideways 1
            return np.array([0.0, 1.0 * factor, 0.0], dtype=np.float32)
        elif action == 5:  # sideways 2
            return np.array([0.0, -1.0 * factor, 0.0], dtype=np.float32)
        elif action == 2:  # rotation 1
            return np.array([0.0, 0.0, 1.0 * rot_factor], dtype=np.float32)
        elif action == 3:  # rotation 2
            return np.array([0.0, 0.0, -1.0 * rot_factor], dtype=np.float32)

    def step(self, action):
        # translate action
        action = self.map_discrete_to_continuous(action)
        _, reward, truncated, terminated, info = self.env.step(action)

        # get observation
        image = self.get_image(self.env, self.camera)
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
