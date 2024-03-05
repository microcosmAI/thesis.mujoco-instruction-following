import json
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path


# TODO naming convention on the functions below,
# TODO simplify code (fewer fcts)


def read_colors(colorset_file_path):
    """Read color names and RGB values from a .json file

    Args:
        json_file (str): Path to .json file containing color information

    Returns:
        list: list of dicts, structured {"name": name, "rgb": rgb value}
    """
    with open(colorset_file_path, "r") as file:
        data = json.load(file)

    colors = []
    for entry in data:
        rgb_values = entry["code"]
        color_name = entry["name"]
        colors.append({"name": color_name, "rgb": rgb_values})

    return colors


def generate_color_list(colorset_file_path, color_amount):
    """Returns the first color_amount colors from a list of colors"""

    colors = read_colors(colorset_file_path)
    return colors[:color_amount]


def read_mujoco_shapes(directory):
    """Reads all .xml files in a directory and returns a list of dicts with the model attribute and the filename
    Does not return default objects (Border, Light, BoxAgent)

    Args:
        directory (str): path to directory containing .xml files

    Returns:
        list: list of dicts, structured {"model": model attribute, "xml_name": filename}
    """

    shapes = []
    for filename in Path(directory).iterdir():
        if filename.suffix == ".xml":
            tree = ET.parse(filename)
            mujoco_element = tree.getroot()
            if mujoco_element is not None:
                model_attribute = mujoco_element.attrib.get("model", "")
                shapes.append({"model": model_attribute, "xml_name": filename.name}) # Example.xml

    # remove default objects
    default_objects = ["Border.xml", "Light.xml", "BoxAgent.xml"]
    shapes = [shape for shape in shapes if shape["xml_name"] not in default_objects]

    return shapes


def generate_shape_list(directory, shape_amount):
    """Returns a list of dicts of the first shape_amount shapes in a directory"""

    shapes = read_mujoco_shapes(directory)
    return shapes[:shape_amount]


def read_instructions_by_type(instr_file_path, instr_type):
    """Reads a json file with "instr_type: instruction" formatted values. Returns a list of dicts of those instructions that are of type instr_type

    Args:
        json_file (str): path to .json file with instructions
        instr_type (str): the type of instruction to filter and return as a list

    Returns:
        list: list of dicts, all instruction elements of instr_type
    """

    with open(instr_file_path, "r") as file:
        data = json.load(file)

    filtered_instructions = []

    for element in data:
        if instr_type in element:
            filtered_instructions.append(element)

    return filtered_instructions


def generate_instr_list_by_type(instr_file_path, instr_type, instr_amount):
    """Returns a list of dicts of the first instr_amount instructions of type instr_type"""

    instructions = read_instructions_by_type(instr_file_path, instr_type)
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
    # Handle instruction lists as one, since each prompt only contains one instruction
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

    # Handle instructions of different types
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

        size_value = ""
        if entry["size"]:
            size_value = entry["size"]["name"]

        color_name = entry["color"]["name"]
        shape_model = entry["shape"]["model"]

        prompt_string = (
            f"{instruction_value} the {size_value} {color_name} {shape_model}"
        )

        # Replace double spaces (for when there is no size modifier)
        prompt_string = prompt_string.replace("  ", " ")

        entry["prompt"] = prompt_string

    return prompt_list


def write_prompts(
    colorset_file_path,
    output_file_path,
    xml_objects_dir_path,
    instr_file_path,
    size_modifier_list,
    color_amount,
    shape_amount,
    size_amount,
    instr_amounts,
    instr_types,
):
    """Generates a file in the prompts_file_path directory based on the given parameters

    Args:
        colorset_file_path (str): path to .json file with color information
        output_file_path (str): path to output file
        xml_objects_dir_path (str): path to directory with .xml files
        instr_file_path (str): path to .json file with instructions
        size_modifier_list (list): list of size modifiers
        color_amount (int): amount of colors
        shape_amount (int): amount of shapes
        size_amount (int): amount of size modifiers
        instr_amounts (list): list of instruction amounts
        instr_types (list): list of instruction types

    Returns:
        none
    """

    # Generate lists of instruction, size, color and shape
    instruction_list = [
        generate_instr_list_by_type(
            instr_file_path=instr_file_path,
            instr_type=instr_type,
            instr_amount=instr_amount,
        )
        for instr_type, instr_amount in zip(instr_types, instr_amounts)
    ]
    size_list = generate_size_modifiers(
        size_amount=size_amount, size_modifier_list=size_modifier_list
    )
    color_list = generate_color_list(
        colorset_file_path=colorset_file_path, color_amount=color_amount
    )
    shape_list = generate_shape_list(
        directory=xml_objects_dir_path, shape_amount=shape_amount
    )

    # Generate list of dicts of all combinations
    prompt_list = generate_prompt_dicts(
        color_list=color_list,
        shape_list=shape_list,
        size_list=size_list,
        instruction_list=instruction_list,
    )

    # Write list of dicts to file
    try:
        print(prompt_list)
        with open(output_file_path, "w") as json_file:
            json.dump(prompt_list, json_file)
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")