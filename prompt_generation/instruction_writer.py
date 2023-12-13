# Generates a .json file of instructions (including which targets would be correct given each instruction) based on which variables are chosen

# TODO load objects, colors, and prompts numbers
import json
import os
import xml.etree.ElementTree as ET


def read_colors(json_file, color_amount):
    with open(json_file, "r") as file:
        data = json.load(file)

    colors = []
    for entry in data[:color_amount]:
        rgb_values = entry["code"]
        color_name = entry["name"]
        colors.append({"name": color_name, "rgb": rgb_values})

    return colors


def read_mujoco_shapes(directory, shape_amount):
    shapes = []
    for filename in os.listdir(directory)[:shape_amount]:
        if filename.endswith(".xml"):
            tree = ET.parse(os.path.join(directory, filename))
            mujoco_element = tree.getroot()
            if mujoco_element is not None:
                model_attribute = mujoco_element.attrib.get("model", "")
                shapes.append({"model": model_attribute, "xml_name": filename})

    return shapes


def read_instructions(json_file, attribute, instr_amount):
    with open(json_file, "r") as file:
        data = json.load(file)

    filtered_instructions = [
        instr for instr in data if instr.get(attribute) is not None
    ][:instr_amount]
    return filtered_instructions


# Test usage:

# test colors
color_file_path = "./data/colors/output_1words_rgb.json"  # TODO path
color_amount = 5  # TODO user input

result = read_colors(color_file_path, color_amount)
print(result)


# test shapes
xml_directory_path = (
    "./data/objects" 
)
shape_amount = 5 
result = read_mujoco_shapes(xml_directory_path, shape_amount)
print(result)

# test instructions
instr_file_path = (
    "./data/instructions/instructions.txt"  
)
attribute_to_filter = "approach"  # Replace with 'approach' or 'avoid'
instr_amount = 5  
result = read_instructions(
    instr_file_path, attribute_to_filter, instr_amount
)
print(result)

# TODO output current amounts
# TODO get user input on how many of each
# TODO check if all info that is needed is there
