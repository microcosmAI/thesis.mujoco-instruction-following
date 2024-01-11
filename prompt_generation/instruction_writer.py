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


def read_instructions_by_type(json_file, instr_type):
    """Reads a json file with "instr_type: instruction" formatted values. Returns a list of dicts of those instructions that are of type instr_type

    Args:
        json_file (str): path to .json file with instructions
        instr_type (str): the type of instruction to filter and return as a list

    Returns:
        list: list of dicts, all instruction elements of instr_type
    """
    with open(json_file, "r") as file:
        data = json.load(file)

    filtered_instructions = []

    for element in data:
        if instr_type in element:
            filtered_instructions.append(element)

    return filtered_instructions


def generate_instr_list_by_type(json_file, instr_type, instr_amount):
    instructions = read_instructions_by_type(json_file, instr_type)
    return instructions[:instr_amount]


def generate_size_modifiers(size_amount, size_modifier_list):
    if size_amount == 1 or size_amount == 0:
        return []
    else:
        return size_modifier_list[:size_amount]


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
        raise TypeError  # TODO implement errors

    # Multiply all non-0 values
    result = 1
    for element in list_lengths:
        if element != 0:
            result *= element
    return result


def generate_prompt_dict(color_list, shape_list, instruction_list, size_list):
    # TODO handle exception for sizes 0 and 1
    # TODO rename such that it is accurate (current output is list of dict)
    instruction_list = [
        {"type": list(instr_dict.keys())[0], "value": list(instr_dict.values())[0]}
        for sublist in instruction_list
        for instr_dict in sublist
    ]

    prompt_list = [
        {"instruction": instruction, "size": size, "color": color, "shape": shape}
        for instruction, size, color, shape in product(
            instruction_list, size_list, color_list, shape_list
        )
    ]


    return prompt_list


def export_to_json(prompt_dicts, output_filepath):
    # TODO add docstring, type for prompts is list of dicts for now
    with open(output_filepath, "w") as json_file:
        json.dump(prompt_dicts, json_file)

    print(f"Prompts have been written to{output_filepath}")


def main():
    # Data to build instructions from
    color_filepath = "./data/colors/output_1words_rgb.json"
    xml_directory_filepath = "./data/objects"
    instr_file_path = "./data/instructions/instructions.txt"
    instr_types = ["approach", "avoid"]
    output_filepath = "./output/prompts.json"

    size_modifier_list = ["large", "small", "huge", "tiny"]
    color_list = read_colors(color_filepath)
    shape_list = read_mujoco_shapes(xml_directory_filepath)
    instr_lists = [
        read_instructions_by_type(instr_file_path, instr_type)
        for instr_type in instr_types
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
        default=3,
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
            default=1,
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
        color_amount=args.color_amount,
        shape_amount=args.shape_amount,
        instruction_amounts=instr_amounts,
        size_amount=args.size_amount,
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
            color_amount=color_amount,
            shape_amount=shape_amount,
            instruction_amounts=instr_amounts,
            size_amount=size_amount,
        )
        print(f"Total Variations: {total_variations}")

        # Update arguments for the next iteration
        args.color_amount = color_amount
        args.shape_amount = shape_amount
        args.instr_amounts = instr_amounts
        args.size_amount = size_amount

        if (
            input(
                "\nDo you want to alter these values? (type no to proceed) (yes/no): "
            ).lower()
            != "yes"
        ):
            break

    # TODO consider what goes here: call the "generate " functions to generate lists of the lengths
    # i want, or just shorten the lists I already have.
    # also, is the format for the instructions okay (being list instead of list of dicts like all the others?)

    combinations = generate_prompt_dict(
        color_list=generate_color_list(
            json_file=color_filepath, color_amount=color_amount
        ),
        shape_list=generate_shape_list(
            directory=xml_directory_filepath, shape_amount=shape_amount
        ),
        size_list=generate_size_modifiers(
            size_amount=size_amount, size_modifier_list=size_modifier_list
        ),
        instruction_list=[
            generate_instr_list_by_type(
                json_file=instr_file_path,
                instr_type=instr_type,
                instr_amount=instr_amount,
            )
            for instr_type, instr_amount in zip(instr_types, instr_amounts)
        ],
    )

    print(combinations) # for debugging

    export_to_json(output_filepath=output_filepath, prompt_dicts=combinations) #TODO change output filepath var name

if __name__ == "__main__":
    main()
