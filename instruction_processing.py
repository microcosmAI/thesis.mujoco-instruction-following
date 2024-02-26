import json
import os
import numpy as np
import torch


def get_word_to_idx(instructions_file_path):
    """
    Gets a dictionary mapping words to their corresponding indices from an instructions file.

    Args:
        instructions_file_path (str): The path to the instructions file.

    Returns:
        dict: A dictionary mapping words to their corresponding indices.
    """
    word_to_idx = {"-": 0} # special token for padding

    with open(instructions_file_path) as json_file:
        data = json.load(json_file)
        for entry in data:
            prompt = entry["prompt"]
            for word in prompt.lower().split(" "):
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)

    return word_to_idx


def get_word_to_idx_from_dir(instructions_dir_path):
    """
    Gets the instruction files from the specified directory path and creates a dictionary mapping each unique word in the file names to its index.

    Args:
        instructions_dir_path (str): The path to the directory containing the instruction files.

    Returns:
        dict: A dictionary mapping each unique word in the file names to its index.
    """
    word_to_idx = {"-": 0} # special token for padding
    filenames = sorted(
        [f for f in os.listdir(instructions_dir_path) if f.endswith(".xml")]
    )

    for filename in filenames:
        prompt = filename.split(".")[0].replace("_", " ")
        for word in prompt.lower().split(" "):
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

    return word_to_idx


def get_max_instruction_length_from_dir(instructions_dir_path):
    """
    Gets the instruction files from the specified directory path and returns the maximum length of the instructions.

    Args:
        instructions_dir_path (str): The path to the directory containing the instruction files.

    Returns:
        int: The maximum length of the instructions.
    """
    max_instruction_length = 0
    filenames = sorted(
        [f for f in os.listdir(instructions_dir_path) if f.endswith(".xml")]
    )

    for filename in filenames:
        prompt = filename.split(".")[0].replace("_", " ")
        prompt_length = len(prompt.split(" "))
        if prompt_length > max_instruction_length:
            max_instruction_length = prompt_length

    return max_instruction_length


def get_word_to_idx_from_curriculum_dir(curriculum_dir_path):
    """
    Gets the instruction files from the specified curriculum directory path and creates a dictionary mapping each unique word in the file names to its index.

    Args:
        curriculum_dir_path (str): The path to the curriculum directory containing the instruction dirs.

    Returns:
        dict: A dictionary mapping each unique word in the file names to its index.
    """
    word_to_idx = {"-": 0} # special token for padding
    level_directories = sorted(
        [
            os.path.join(curriculum_dir_path, d)
            for d in os.listdir(curriculum_dir_path)
            if os.path.isdir(os.path.join(curriculum_dir_path, d))
        ]
    )

    for level_dir in level_directories:
        word_to_idx.update(get_word_to_idx_from_dir(level_dir))

    return word_to_idx


def get_max_instruction_length_from_curriculum_dir(curriculum_dir_path):
    """
    Gets the instruction files from the specified curriculum directory path and returns the maximum length of the instructions.

    Args:
        curriculum_dir_path (str): The path to the curriculum directory containing the instruction dirs.

    Returns:
        int: The maximum length of the instructions.
    """
    max_instruction_length = 0
    level_directories = sorted(
        [
            os.path.join(curriculum_dir_path, d)
            for d in os.listdir(curriculum_dir_path)
            if os.path.isdir(os.path.join(curriculum_dir_path, d))
        ]
    )

    for level_dir in level_directories:
        max_instruction_length = max(
            max_instruction_length, get_max_instruction_length_from_dir(level_dir)
        )

    return max_instruction_length


def get_instruction_idx(instruction, word_to_idx, max_instr_length):
    """Get the idx for a single instruction based on the word_to_idx dictionary."""
    instruction_idx = []
    for word in instruction.split(" "):
        instruction_idx.append(word_to_idx[word])
    # Pad the instruction to the maximum instruction length using 0 as special token
    pad_length = max_instr_length - len(instruction_idx)
    if pad_length > 0:
        instruction_idx += [0] * pad_length 
    instruction_idx = np.array(instruction_idx)
    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)
    return instruction_idx