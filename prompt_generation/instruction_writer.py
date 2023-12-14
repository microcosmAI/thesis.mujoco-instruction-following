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


def generate_size_modifiers(size_amount):
    size_modifiers = ["large", "small", "huge", "tiny"]

    if size_amount == 1:
        return []
    else:
        return size_modifiers[:size_amount]


# TODO output current amounts
# TODO get user input on how many of each
# TODO check if all info that is needed is there
