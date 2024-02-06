import json
import os

def get_word_to_idx(instructions_file_path):
    """
    Gets a dictionary mapping words to their corresponding indices from an instructions file.

    Args:
        instructions_file_path (str): The path to the instructions file.

    Returns:
        dict: A dictionary mapping words to their corresponding indices.
    """
    word_to_idx = {}

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
    word_to_idx = {}
    filenames = sorted([f for f in os.listdir(instructions_dir_path) if f.endswith(".xml")])

    for filename in filenames:
        prompt = filename.split(".")[0].replace("_", " ")
        for word in prompt.lower().split(" "):
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

    return word_to_idx
