# Generates a .json file of instructions (including which targets would be correct given each instruction) based on which variables are chosen

# TODO load objects, colors, and prompts numbers
import json
import os
import xml.etree.ElementTree as ET
import argparse


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


def read_instructions(json_file, attribute):
    with open(json_file, "r") as file:
        data = json.load(file)

    filtered_instructions = [
        instr for instr in data if instr.get(attribute) is not None
    ]
    return filtered_instructions


def generate_instr_list(json_file, attribute, instr_amount):
    instructions = read_instructions(json_file, attribute)
    return instructions[instr_amount]


def generate_size_modifiers(size_amount, size_modifiers):
    if size_amount == 1:
        return []
    else:
        return size_modifiers[:size_amount]


def calculate_total_variations(*args):
    """Multiply all args to get the amount of total possible variations

    Returns:
        int: All args, multiplied
    """
    result = 1
    for num in args:
        if num != None:
            result *= num
    return result


def main():
    # Data to build instructions from
    color_file_path = "./data/colors/output_1words_rgb.json"
    xml_directory_path = "./data/objects"
    instr_file_path = "./data/instructions/instructions.txt"
    attributes = [
        "approach",
        "avoid",
    ]  # TODO naming: check if attribute is the right word here
    size_modifiers = ["large", "small", "huge", "tiny"]

    # Metadata for CLI
    max_color_amount = len(read_colors(color_file_path))
    max_shape_amount = len(read_mujoco_shapes(xml_directory_path))
    max_instr_amounts = [
        len(read_instructions(instr_file_path, attribute)) for attribute in attributes
    ]
    max_size_amount = len(size_modifiers)

    # Parser
    parser = argparse.ArgumentParser(
        description="Process instructions and count occurrences."
    )

    # Add command-line arguments for user input
    parser.add_argument(
        "--color_amount",
        type=int,
        choices=range(1, max_color_amount + 1),
        help="Number of colors",
    )
    parser.add_argument(
        "--shape_amount",
        type=int,
        choices=range(1, max_shape_amount + 1),
        help="Number of shapes",
    )

    # Add dynamic arguments based on max_instr_amounts and attributes
    for i, max_instr_amount in enumerate(max_instr_amounts):
        attribute_name = attributes[i] + "_instr_amount"
        parser.add_argument(
            f"--{attribute_name}",
            type=int,
            choices=range(1, max_instr_amount + 1),
            default=[],
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
        args.color_amount, args.shape_amount, *instr_amounts, args.size_amount
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
            attribute_name = attributes[i] + "_instr_amount"
            current_value = getattr(args, attribute_name)
            new_value = int(
                    input(f"Amount of instructions of type {attributes[i]} ({current_value}): ")
                    or current_value,
                )
            instr_amounts.append(new_value)

        print(instr_amounts)

        size_amount = int(
            input(f"Size Amount ({args.size_amount}): ") or args.size_amount
        )

        # Calculate total variations
        total_variations = calculate_total_variations(
            color_amount, shape_amount, *instr_amounts, size_amount
        )
        print(f"Total Variations: {total_variations}")

        # Update arguments for the next iteration
        args.color_amount = color_amount
        args.shape_amount = shape_amount
        args.instr_amounts = instr_amounts
        args.size_amount = size_amount

        if input("\nDo you want to continue? (yes/no): ").lower() != "yes":
            break


if __name__ == "__main__":
    main()
