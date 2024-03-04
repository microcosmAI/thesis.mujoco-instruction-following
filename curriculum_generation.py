from pathlib import Path
import shutil
import random

from colorset_generation import colorset_writer
from prompt_generation import prompt_writer
from xml_generation import xml_writer

# TODO write test levels


def get_default_level_parameters(level_amount):
    """Return default level parameters as a dict of lists of ints

    Args:
        level_amount (int): total amount of levels in the curriculum

    Returns:
        dict: dict of lists of ints of length level_amount
    """
    color_amounts = [2, 2, 3, 3, 4, 5, 6, 6, 6][:level_amount]
    shape_amounts = [1, 2, 2, 3, 3, 4, 5, 6, 6][:level_amount]
    size_amounts = [2, 2, 2, 2, 2, 2, 2, 2, 2][:level_amount]
    instr_type_amounts = 2
    instr_amounts = [
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5][:level_amount],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0][:level_amount], # NOTE reward functions for non-approach type instructions are not implemented
    ][:instr_type_amounts]

    return {
        "color_amounts": color_amounts,
        "shape_amounts": shape_amounts,
        "size_amounts": size_amounts,
        "instr_type_amounts": instr_type_amounts,
        "instr_amounts": instr_amounts,
    }


# Function that takes the generated levels of the highest level, and removes 1/3 of the content of each, then returns all the removed levels
def pull_test_set(curriculum_dir_path, test_set_dir_path, test_set_ratio):
    """Move a random subset of the highest level of the curriculum to a test set directory

    Args:
        curriculum_dir_path (Path): path to the curriculum directory
        test_set_ratio (float): ratio of test stages to total stages - taken from the highest level

    Returns:
        None
    """

    # create test set directory
    if test_set_dir_path.exists():
        shutil.rmtree(test_set_dir_path)
    test_set_dir_path.mkdir()

    highest_level_dir_path = curriculum_dir_path / sorted(curriculum_dir_path.iterdir())[-1].name

    # get all xml files from the highest level
    highest_level_files = [
        file.name for file in highest_level_dir_path.iterdir() if file.name.endswith(".xml")
    ]
    amount_of_files_to_remove = int(len(highest_level_files) * test_set_ratio)

    # pull random files for the test set (get corresponding .json files too)
    random.shuffle(highest_level_files)
    test_set_files = highest_level_files[:amount_of_files_to_remove]
    test_set_files += [file.replace(".xml", ".json") for file in test_set_files]

    # move files
    for file in test_set_files:
        shutil.move(
            highest_level_dir_path / file,
            test_set_dir_path / file,
        )

    # make sure no files from the test set remain in the training set
    for level_dir in curriculum_dir_path.iterdir():
        level_dir_path = curriculum_dir_path / level_dir.name
        for file in test_set_files:
            if file in [f.name for f in level_dir_path.iterdir()]:
                (level_dir_path / file).unlink() # TODO test if levels work if you just remove some files

    return None


def main():
    cwd = Path.cwd()
    xkcd_dataset_file_path = cwd / "data" / "color-data" / "rgb.txt"
    colorset_dir_path = cwd / "data" / "color-data"
    prompts_file_path = cwd / "data" / "prompt-data" / "prompts.json"
    curriculum_dir_path = cwd / "data" / "curriculum"
    test_set_dir_path = cwd / "data" / "test-set"
    xml_object_dir_path = cwd / "data" / "objects"
    instr_file_path = cwd / "data" / "prompt-data" / "instructions.txt"

    instr_types = ["approach", "avoid"]
    size_modifiers = [
        {"name": "large", "factor": 2},
        {"name": "small", "factor": 0.5},
        {"name": "huge", "factor": 4},
        {"name": "tiny", "factor": 0.25},
    ]
    test_set_ratio = (
        0.33  # ratio of test stages to total stages - taken from the highest level
    )

    level_amount = 3
    params = get_default_level_parameters(level_amount)

    colorset_writer.generate_colorset(
        max_words=1,  # word amount per color (e.g. "green" vs "dark green")
        output_format="json",
        color_format="rgba",
        dataset_path=xkcd_dataset_file_path,
        output_dir_path=colorset_dir_path,
    )

    colorset_file_path = colorset_dir_path / "colors.json"

    # generate one set of prompts and their corresponding xmls for each level
    # generate one set of prompts and their corresponding xmls for each level
    for level_number in range(level_amount):
        # generate a new directory for each level
        level_dir_path = curriculum_dir_path / f"level{level_number}"
        if level_dir_path.exists():
            shutil.rmtree(level_dir_path)
        level_dir_path.mkdir()

        prompts_file_path = level_dir_path / f"prompts-lvl{level_number}.json"

        instr_amounts = [
            instr_amount[level_number] for instr_amount in params["instr_amounts"]
        ]

        # generate prompts
        prompt_writer.write_prompts(
            colorset_file_path=colorset_file_path,
            output_file_path=prompts_file_path,
            xml_objects_dir_path=xml_object_dir_path,
            instr_file_path=instr_file_path,
            color_amount=params["color_amounts"][level_number],
            shape_amount=params["shape_amounts"][level_number],
            size_amount=params["size_amounts"][level_number],
            instr_amounts=instr_amounts,
            size_modifier_list=size_modifiers,
            instr_types=instr_types,  # TODO move to params
        )

        # generate xmls
        xml_writer.write_environments(
            prompts_file_path=prompts_file_path,
            yml_output_dir_path=level_dir_path,
            xml_output_dir_path=level_dir_path,
            xml_object_dir_path=xml_object_dir_path,
            colorset_file_path=colorset_file_path,
            color_amount=params["color_amounts"][level_number],
            size_modifier_list=size_modifiers,
            size_amount=params["size_amounts"][level_number],
        )

    # generate test set
    pull_test_set(
        curriculum_dir_path=curriculum_dir_path,
        test_set_dir_path=test_set_dir_path,
        test_set_ratio=test_set_ratio,
    )


if __name__ == "__main__":
    main()
