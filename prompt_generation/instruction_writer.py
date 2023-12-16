# Generates a .json file of instructions (including which targets would be correct given each instruction) based on which variables are chosen

# TODO load objects, colors, and prompts numbers
# TODO make sure required values are set as required params

import json
import os
import xml.etree.ElementTree as ET
import argparse
from itertools import product


def read_colors(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)

    colors = []
    for entry in data:
        rgb_values = entry["code"]
        color_name = entry["name"]
        colors.append({"name": color_name, "rgb": rgb_values})

    return colors


def generate_color_list(json_file, color_amount):
    colors = read_colors(json_file)
    return colors[:color_amount]


def read_mujoco_shapes(directory):
    shapes = []
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            tree = ET.parse(os.path.join(directory, filename))
            mujoco_element = tree.getroot()
            if mujoco_element is not None:
                model_attribute = mujoco_element.attrib.get("model", "")
                shapes.append({"model": model_attribute, "xml_name": filename})

    return shapes


def generate_shape_list(directory, shape_amount):
    shapes = read_mujoco_shapes(directory)
    return shapes[:shape_amount]


def read_instructions(json_file, instr_type):
    """Reads a json file with "instr_type: instruction" formatted values. Returns a list of those instructions that are of instr_type

    Args:
        json_file (str): path to .json file with instructions
        instr_type (str): the type of instruction to filter and return as a list

    Returns:
        list: all instruction elements of instr_type
    """    
    with open(json_file, "r") as file:
        data = json.load(file)

    filtered_instructions = []
    for element in data:
        if instr_type in element:
            filtered_instructions.append(element[instr_type])

    return filtered_instructions


def generate_instr_list(json_file, instr_type, instr_amount):
    instructions = read_instructions(json_file, instr_type)
    return instructions[instr_amount]


def generate_size_modifiers(size_amount, size_modifier_list):
    if size_amount == 1:
        return []
    else:
        return size_modifier_list[:size_amount]


def calculate_total_variations_old(
    shape_list, *instruction_lists, color_list=None, size_list=None
):
    """Multiply the length of all lists to get the amount of total possible variations

    Args:
        shape_list (list): list of shapes for prompt generation
        *instruction_lists (list): list of lists, each containing instructions of one type
        color_list (list): list of colors for prompt generation
        size_list (list): list of sizes for prompt generation

    Returns:
        int: Length of all args (if non-zero), multiplied (*instruction_lists handled like a single list)
    """

    list_lengths = [color_list, shape_list, size_list]

    # Handle instruction lists as one, since each prompt only contains one instruction
    for list in instruction_lists:
        if list != None:
            list_lengths.append(len(list))

    # Multiply all values to get amount of variations, ignoring 0-values
    result = 1
    for element in list_lengths:
        if element != None:
            if len(element != 0):
                result *= len(element)
    return result


def calculate_total_variations(
    shape_amount, instruction_amounts, color_amount, size_amount
):
    # TODO update docstring
    """Multiply the length of all lists to get the amount of total possible variations

    Args: OUTDATED
        shape_list (list): list of shapes for prompt generation
        *instruction_lists (list): list of lists, each containing instructions of one type
        color_list (list): list of colors for prompt generation
        size_list (list): list of sizes for prompt generation

    Returns:
        int: Length of all args (if non-zero), multiplied (*instruction_lists handled like a single list)
    """

    list_lengths = [shape_amount, color_amount, size_amount]
    # Handle instruction lists as one, since each prompt only contains one instruction
    if type(instruction_amounts) == list:
        instr_length = 0
        for i in instruction_amounts:
            instr_length += i
        list_lengths.append(instr_length)
    elif type(instruction_amounts) == int:
        list_lengths.append(instruction_amounts)
    else:
        raise TypeError #TODO implement errors

    # Multiply all non-0 values
    result = 1
    for element in list_lengths:
        if element != 0:
            result *= element
    return result


def generate_prompt_dict(
    color_list, shape_list, *instruction_lists, size_list=None
):
    prompt_dict = {}

    # Create a list containing all combinations of elements from the input lists
    combinations = calculate_total_variations(
        color_list, shape_list, *instruction_lists, size_list
    )

    for combination in combinations:
        color, shape = combination[:2]
        instructions = {"color": color, "shape": shape}

        # Add size to the instruction if size_list is provided
        if size_list:
            instructions["size"] = size_list[color_list.index(color)]

        # Add additional instructions from the rest of the combination
        for idx, instruction_list in enumerate(combination[2:]):
            instructions[f"instruction_list_{idx + 1}"] = instruction_list

        # Create a unique key for each combination
        key = tuple(combination)
        prompt_dict[key] = instructions

    return prompt_dict


def main():
    # Data to build instructions from
    color_file_path = "./data/colors/output_1words_rgb.json"
    xml_directory_path = "./data/objects"
    instr_file_path = "./data/instructions/instructions.txt"
    instr_types = ["approach", "avoid"]

    size_modifier_list = ["large", "small", "huge", "tiny"]
    color_list = read_colors(color_file_path)
    shape_list = read_mujoco_shapes(xml_directory_path)
    instr_lists = [
        read_instructions(instr_file_path, instr_type) for instr_type in instr_types
    ]

    # Metadata for CLI
    max_color_amount = len(color_list)
    max_shape_amount = len(shape_list)
    max_instr_amounts = [len(instr_list) for instr_list in instr_lists]
    max_size_amount = len(size_modifier_list)

    # Parser
    parser = argparse.ArgumentParser(
        description="Process instructions and count occurrences."
    )

    # Add command-line arguments for user input
    parser.add_argument(
        "--color_amount",
        type=int,
        choices=range(1, max_color_amount + 1),
        default=2,
        help="Number of colors",
    )
    parser.add_argument(
        "--shape_amount",
        type=int,
        choices=range(1, max_shape_amount + 1),
        default=2,
        help="Number of shapes",
    )

    # Add dynamic arguments based on max_instr_amounts and instr_types
    for i, max_instr_amount in enumerate(max_instr_amounts):
        attribute_name = instr_types[i] + "_instr_amount"
        parser.add_argument(
            f"--{attribute_name}",
            type=int,
            choices=range(1, max_instr_amount + 1),
            default=[1],
            nargs="+",
            help=f"Instruction amounts for {attribute_name}",
        )


    parser.add_argument(
        "--size_amount",
        type=int,
        choices=range(1, max_size_amount + 1),
        default=1,
        help="Size amount",
    )

    args = parser.parse_args()

    # Initial calculation
    instr_amounts = max_instr_amounts
    total_variations = calculate_total_variations(
        color_amount=args.color_amount, shape_amount=args.shape_amount, instruction_amounts=instr_amounts, size_amount=args.size_amount
    )
    print(f"Initial Total Variations: {total_variations}")

    while True:
        # Ask user for input
        print('\nEnter new values (type "exit" to quit):')
        color_amount = int(
            input(f"Color Amount ({args.color_amount}): ") or args.color_amount
        )
        shape_amount = int(
            input(f"Shape Amount ({args.shape_amount}): ") or args.shape_amount
        )
        instr_amounts = []
        for i, max_instr_amount in enumerate(max_instr_amounts):
            attribute_name = instr_types[i] + "_instr_amount"
            current_value = getattr(args, attribute_name)
            new_value = int(
                input(
                    f"Amount of instructions of type {instr_types[i]} ({current_value}): "
                )
                or current_value,
            )
            instr_amounts.append(new_value)

        size_amount = int(
            input(f"Size Amount ({args.size_amount}): ") or args.size_amount
        )

        # Calculate total variations
        total_variations = calculate_total_variations(
            color_amount=color_amount, shape_amount=shape_amount, instruction_amounts=instr_amounts, size_amount=size_amount
        )
        print(f"Total Variations: {total_variations}")

        # Update arguments for the next iteration
        args.color_amount = color_amount
        args.shape_amount = shape_amount
        args.instr_amounts = instr_amounts
        args.size_amount = size_amount

        if input("\nDo you want to continue? (yes/no): ").lower() != "yes":
            break

    combinations = generate_prompt_dict(
        color_list=color_list,
        shape_list=shape_list,
        size_list=size_modifier_list,
        *instr_lists,
    )

    print(combinations)


if __name__ == "__main__":
    main()
