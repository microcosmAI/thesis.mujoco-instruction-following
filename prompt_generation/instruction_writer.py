import json
import os
import xml.etree.ElementTree as ET
import argparse
from itertools import product


def read_colors(json_file):
    """Read color names and RGB values from a .json file

    Args:
        json_file (str): Path to .json file containing color information

    Returns:
        list: list of dicts, structured {"name": name, "rgb": rgb value}
    """
    with open(json_file, "r") as file:
        data = json.load(file)

    colors = []
    for entry in data:
        rgb_values = entry["code"]
        color_name = entry["name"]
        colors.append({"name": color_name, "rgb": rgb_values})

    return colors


def generate_color_list(json_file, color_amount):
    """Returns the first color_amount colors from a list of colors"""

    colors = read_colors(json_file)
    return colors[:color_amount]


def read_mujoco_shapes(directory):
    """Reads all .xml files in a directory and returns a list of dicts with the model attribute and the filename

    Args:
        directory (str): path to directory containing .xml files

    Returns:
        list: list of dicts, structured {"model": model attribute, "xml_name": filename}
    """

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
    """Returns a list of dicts of the first shape_amount shapes in a directory"""

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
    """Returns a list of dicts of the first instr_amount instructions of type instr_type"""

    instructions = read_instructions_by_type(json_file, instr_type)
    return instructions[:instr_amount]


def generate_size_modifiers(size_amount, size_modifier_list):
    """Returns a list of size modifiers of length size_amount"""

    if size_amount == 1 or size_amount == 0:
        return [""]
    else:
        return size_modifier_list[:size_amount]


def calculate_total_variations(
    shape_amount, instruction_amounts, color_amount, size_amount
):
    """Calculates the total amount of variations that can be made from the given parameters

    Args:
        shape_amount (int): amount of shapes
        instruction_amounts (int or list): amount of instructions (can be list of ints if multiple instruction types are used)
        color_amount (int): amount of colors
        size_amount (int): amount of size modifiers

    Returns:
        int: total amount of variations
    """

    list_lengths = [shape_amount, color_amount, size_amount]
    # Handle all instruction lists as one, since each prompt only contains one instruction
    if type(instruction_amounts) == list:
        instr_length = 0
        for i in instruction_amounts:
            instr_length += i
        list_lengths.append(instr_length)
    elif type(instruction_amounts) == int:
        list_lengths.append(instruction_amounts)
    else:
        raise TypeError(
            f"Expected instruction_amounts to be int or list, got {type(instruction_amounts)}"
        )

    # Multiply all non-0 values
    result = 1
    for element in list_lengths:
        if element != 0:
            result *= element
    return result


def generate_prompt_dicts(color_list, shape_list, instruction_list, size_list):
    """Generates a list of dicts of all possible combinations of the given lists

    Args:
        color_list (list): list of color dicts
        shape_list (list): list of shape dicts
        instruction_list (list): list of instruction dicts
        size_list (list): list of size modifier strings

    Returns:
        list: list of dicts, structured {"instruction": instruction, "size": size, "color": color, "shape": shape, "prompt": prompt_string}
    """

    # Generate list of dicts of all combinations
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

    # Generate prompt string for each dict, add prompt string to list
    for entry in prompt_list:
        instruction_value = entry["instruction"]["value"]
        size_value = entry["size"]
        color_name = entry["color"]["name"]
        shape_model = entry["shape"]["model"]

        prompt_string = (
            f"{instruction_value} the {size_value} {color_name} {shape_model}"
        )

        # Replace double spaces (for when there is no size modifier)
        prompt_string = prompt_string.replace("  ", " ")

        entry["prompt"] = prompt_string

    return prompt_list


def export_to_json(prompt_dicts, output_filepath):
    """Write a list of prompt dicts to a .json file"""

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

    combinations = generate_prompt_dicts(
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

    export_to_json(output_filepath=output_filepath, prompt_dicts=combinations)


if __name__ == "__main__":
    main()
