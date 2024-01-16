# Script that calls the following scripts in order:
#  1. ./colorset_generation/colorset_writer.py
#  2. ./prompt_generation/prompt_writer.py
#  3. ./xml_generation/xml_writer.py
#
# This script should be structured as follows:
#  1. Import all necessary modules
#  2. Ask user whether they want default or custom curriculum
#  3. If default, call curriculum generation function with default parameters
#  4. If custom, ask user for parameters:
#     a. Generate colorset?
#      i. If yes, call colorset generation function with user parameters
#     b. Generate prompt?
#      i. If yes, call prompt generation function with user parameters
#     c. Generate XML?
#      i. If yes, call XML generation function with user parameters
#     d. Generate curriculum?
#      i. If yes, call curriculum generation function with user parameters
#     (the steps above should be called such that if one is yes, the ones that follow are also yes)
#  5. Call curriculum generation function with user parameters
#  6. Print confirmation message
#
# The curriculum generation script works as follows:
#  Depending on the parameters, create total_levels amount of folders
#  For each folder, do step 4 from above if the user wants custom parameters, 3 otherwise

import os
import sys
import xml.etree.ElementTree as ET
import argparse


# Import all necessary modules
from colorset_generation import colorset_writer
from prompt_generation import prompt_writer
from xml_generation import xml_writer


# Function that takes level amount, and returns a dict of lists of ints, where the keys are the parameters for the prompt writing
def get_default_level_parameters(level_amount):
    """Return default level parameters as a dict of lists of ints

    Args:
        level_amount (int): total amount of levels in the curriculum

    Returns:
        dict: dict of lists of ints of length level_amount
    """
    color_amounts = [2, 2, 2, 2, 2, 2, 2, 2, 2][:level_amount]
    shape_amounts = [1, 1, 1, 1, 1, 1, 1, 1, 1][:level_amount]
    size_amounts = [1, 1, 1, 1, 1, 1, 1, 1, 1][:level_amount]
    instr_type_amounts = 2  # Must be the max amount of instr types in any level
    instr_amounts = [
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2][:level_amount],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1][:level_amount],
    ][:instr_type_amounts]

    return {
        "color_amounts": color_amounts,
        "shape_amounts": shape_amounts,
        "size_amounts": size_amounts,
        "instr_type_amounts": instr_type_amounts,
        "instr_amounts": instr_amounts,
    }


def get_custom_level_parameters():
    # TODO
    return None


def generate_level(
    level_number, color_amount, shape_amount, size_amount, instr_amounts, output_path
):
    print(f"Generating level {level_number}...")
    print("color_amount:", color_amount)
    print("shape_amount:", shape_amount)
    print("size_amount:", size_amount)
    print("instr_amounts:", instr_amounts)


def generate_curriculum(output_path, level_parameters):
    # iterate over level_parameters, which is a dict of lists of ints. The keys are the parameters for the generate_level function.
    # within this function, call generate_level with the parameters from the dict

    for (
        level_number,
        color_amount,
        shape_amount,
        size_amount,
        instr_amounts,
    ) in level_parameters:
        generate_level(
            level_number,
            color_amount,
            shape_amount,
            size_amount,
            instr_amounts,
            output_path,
        )


# Function that takes the generated levels of the highest level, and removes 1/3 of the content of each, then returns all the removed levels
def get_test_levels(levels):
    # TODO
    return None


def move_test_levels(test_levels, output_path):
    # TODO move test levels to different folder
    return None


def main():
    xkcd_dataset_file_path = os.path.join(os.getcwd(), "data", "xkcd_color_dataset.txt")
    color_file_path = os.path.join(os.getcwd(), "data", "colors.json")
    prompts_file_path = os.path.join(os.getcwd(), "data", "prompts.json")
    curriculum_dir_path = os.path.join(
        os.getcwd(), "data", "curriculum"
    )  # TODO unify naming convention (dir vs file)

    print(xkcd_dataset_file_path)
    # TODO move all files to their respective directories
    
    level_amount = 5
    params = get_default_level_parameters(level_amount)

    colorset = colorset_writer.generate_colorset(
        max_words=1, # word amount per color (e.g. "green" vs "dark green")
        output_format="json",
        color_format="rgba",
        dataset_path=xkcd_dataset_file_path,
        output_path=color_file_path,
    )

    # generate one set of prompts and their corresponding xmls for each level
    # the amount of levels 
    for level_number in range(level_amount):
        # generate a new directory for each level
        level_dir_path = os.path.join(curriculum_dir_path, f"level_{level_number}")
        os.mkdir(level_dir_path)

        # generate prompts
        prompt_writer.generate_prompts(
            colorset=colorset,
            prompts_file_path=prompts_file_path,
            level_number=level_number,
            color_amount=params["color_amounts"][level_number],
            shape_amount=params["shape_amounts"][level_number],
            size_amount=params["size_amounts"][level_number],
            instr_amounts=params["instr_amounts"],
        )

        # generate xmls
        xml_writer.generate_xmls(
            prompts_file_path=prompts_file_path,
            xml_output_path=level_dir_path,
            xml_object_path=os.path.join(os.getcwd(), "data", "objects"),
        )



    generate_curriculum(curriculum_dir_path, params)

    """
    # Ask user whether they want default or custom curriculum
    # Use argpase for this:
    #  -d or --default for default
    #  -c or --custom for custom
    #  -h or --help for help
    parser = argparse.ArgumentParser(description='Generate a curriculum.')
    parser.add_argument('-d', '--default', action='store_true', help='Generate a default curriculum.')
    parser.add_argument('-c', '--custom', action='store_true', help='Generate a custom curriculum.')

    args = parser.parse_args()

    # If default, call curriculum generation function with default parameters
    if args.default:
        # TODO
        pass
    # If custom, ask user for parameters:
    elif args.custom:
        get_custom_prompt_parameters()

    """


if __name__ == "__main__":
    main()
