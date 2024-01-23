# Goal here is to build a yml, call PITA on it, then modify the resulting xml to fit my experiment.
# This script should only generate a single set of xml files based on the json file it gets.

# The yml should be as follows:
#   - at least one object of the same type as the target object, but of different color OR at least one object of the same color as the target object, but of different type
#   - the color choices should be from the colorset pool of rgb values
#
# Then, we will change the color of the objects according to how we need them to be:
#   - iterate over the objects. note: potentially shuffle them, to avoid biases
#   - the first one of target type is the target object:
#       - give it the correct color, and rename it TARGET
#   - all others get a random color such that the conditions are met (see above)

# TODO decide on yml vs yml

import os
import yaml
import json
import numpy as np
from pita_algorithm.pita import PITA
import re

def write_yml_entry(entry, yml_output_dir_path, object_pool):
    """Write a single yml file based on one entry in the json file

    Args:
        entry (dict): Entry describing the target object and prompt
        yml_output_dir_path (str): Path to output directory
        object_pool (list): List of all object types that occur in the json file

    Raises:
        ValueError: If no object shape is found in the entry
    """

    # Define the fixed structure for the yml file
    yml_data = {

        # Environment size is fixed at 100x100
        "Environment": {
            "size_range": [100, 100],
            "Style": [{"pretty_mode": False}],
            "Borders": [
                {"xml_name": "Border.xml"},
                {"place": True},
                {"tags": ["Border"]},
            ],
            # Placeholder object because PITA needs at least one object in the environment to work (in its current version)
            "Objects": {
                "Placeholder": [
                    {"xml_name": "Box.xml"},
                    {"amount": 1},
                    {"coordinates": [[1, 1, -30]]}, # under the floor
                    {"tags": ["Placeholder"]},
                ],
            },
        },

        # Placing the objects in one half, the agent in the other half by splitting into two areas
        "Areas": {
            "Area1": {
                "Objects": {
                    "Agent": [
                        {"xml_name": "BoxAgent.xml"},
                        {"amount": [1, 1]},
                        {"z_rotation_range": [-180, 180]},
                        {"tags": ["Agent"]},
                    ],
                },
            },
            "Area2": {
                "Objects": {},
            },
        },
    }

    # Add target object and four distractors
    for i in range(0, 4):
        # Note: xml_name refers to the name of the xml object that gets loaded (providing the object shape)
        if i == 0:
            # Add one target object based on the json entry, with target shape
            if "shape" in entry and "xml_name" in entry["shape"]:
                xml_name = entry["shape"]["xml_name"]
            else:
                raise ValueError("No shape specified for target object")

            obj_structure = [
                {"xml_name": f"{xml_name}"},
                {
                    "amount": [1, 1]
                },  # Needs to be a range for PITA to generate a random position
                {"z_rotation_range": [-180, 180]},
                {"tags": ["Target"]},
            ]

        elif i == 1:
            # Gurarantee one distractor of the same shape as the target object (will get a different color later)
            if "shape" in entry and "xml_name" in entry["shape"]:
                xml_name = entry["shape"]["xml_name"]
            else:
                raise ValueError("No shape specified for target object")

            obj_structure = [
                {"xml_name": f"{xml_name}"},
                {"amount": [1, 1]},
                {"z_rotation_range": [-180, 180]},
                {"tags": ["Distractor"]},
            ]

        else:
            if "shape" in entry and "xml_name" in entry["shape"]:
                xml_name = entry["shape"]["xml_name"]

            obj_structure = [
                {"xml_name": f"{xml_name}"},
                {"amount": [1, 1]},
                {"z_rotation_range": [-180, 180]},
                {"tags": ["Distractor"]},
            ]

            # Add distractors of different shape using PITAs object randomization
            non_target_shapes = [shape for shape in object_pool if shape != xml_name]

            if len(non_target_shapes) > 0:
                obj_structure.append({"asset_pool": non_target_shapes})

        obj_name = xml_name.split(".")[0]
        obj_name = f"{obj_name}{i}"  # avoid duplicates
        yml_data["Areas"]["Area2"]["Objects"][obj_name] = obj_structure

    # Get filename from prompt
    entry_name = entry["prompt"].replace(" ", "_").lower()
    yml_output_file_path = os.path.join(yml_output_dir_path, f"{entry_name}.yml")

    os.makedirs(os.path.dirname(yml_output_file_path), exist_ok=True)

    # with open(yml_output_path, 'w') as yml_file:
    #    yaml.dump(yml_data, yml_file, default_flow_style=None, indent=2)

    # Convert data to yml
    yml_str = yaml.dump(yml_data, default_flow_style=None, indent=2)

    # Alter the yml string formatting to the current expected format for PITA (subject to change)
    yml_str = yml_str.replace("{", "").replace("}", "").replace("'", "")

    # Write the modified yml string to file
    with open(yml_output_file_path, "w") as yml_file:
        yml_file.write(yml_str)


def write_xml_entry(
    entry,
    yml_output_dir_path,
    xml_object_dir_path,
    xml_output_dir_path,
    colorset_file_path,
    color_amount,
):
    """Generates a single xml file based on a single yml file, using details from the json file containing the prompts"""

    # get yml/xml filenames from json entry
    yml_file_path = os.path.join(
        yml_output_dir_path, entry["prompt"].replace(" ", "_").lower() + ".yml"
    )
    xml_file_path = os.path.join(
        xml_output_dir_path, entry["prompt"].replace(" ", "_").lower() + ".xml"
    )

    xml_object_dir_path = os.path.join(xml_object_dir_path)

    # call PITA
    PITA().run(
        random_seed=None,
        config_path=yml_file_path,
        xml_dir=xml_object_dir_path,
        export_path=xml_file_path.removesuffix(
            ".xml"
        ),  # TODO test if it works with suffix
        plot=False,
    )

    # TODO change the colors of the objects in the xml file according to my experiment
    modify_xml(xml_file_path, entry, colorset_file_path, color_amount)


def get_object_pool(data):
    """Returns a list of every object type (=xml_name) that occurs in data, minus default objects like borders and agents"""

    # get all objects
    object_pool = []
    for entry in data:
        if "shape" in entry and "xml_name" in entry["shape"]:
            xml_name = entry["shape"]["xml_name"]
            if xml_name not in object_pool:
                object_pool.append(entry["shape"]["xml_name"])
        else:
            raise ValueError("No shape specified for target object")

    # remove default objects
    default_objects = ["BoxAgent.xml", "Border.xml"]
    object_pool = [obj for obj in object_pool if obj not in default_objects]

    return object_pool


def modify_xml(xml_file_path, entry, colorset_file_path, color_amount):
    """
    Modifies the xml file at xml_file_path based on the entry in the json file.

    The modifications are as follows:
        - the target object gets the target color
        - the distractor objects get a random color from the colorset
        - the placeholder objects get deleted

    This function replaces functionality that PITA doesn't have yet.
    Should PITA be able to assign colors to objects, this function can be replaced.

    Args:
        xml_file_path (str): path to xml file
        entry (dict): entry from json file
        colorset_file_path (str): path to colorset file
        color_amount (int): amount of colors to use from colorset file
    """

    # set target color rgba
    target_color = entry["color"]
    json_file_path = xml_file_path.replace(".xml", ".json")

    # get colorset, without target color and with right amount of colors
    with open(colorset_file_path, "r") as f:
        colorset = json.load(f)

    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    with open(xml_file_path, "r") as f:
        xml_data = f.readlines()

    colorset = colorset[:color_amount]
    colorset = [color for color in colorset if color != target_color]

    target_positions = {}  # format: {position: color} (position is a tuple)
    distractor_positions = {}
    placeholder_positions = []

    # modify the json file, store positions as object ID for modifying the xml file
    for key in json_data:
        # Objects with tags can be either:
        #  - in "environment" key, where they are one level deep
        #  - in "areas" key, where they are two levels deep

        if key == "environment":
            keys_to_delete = []
            for k, v in json_data[key]["objects"].items():
                if "Target" in v["tags"]:
                    v["color"] = target_color["rgb"]
                    target_positions[tuple(v["position"])] = v["color"]       
                if "Distractor" in v["tags"]:
                    v["color"] = np.random.choice(colorset)["code"]
                    distractor_positions[tuple(v["position"])] = v["color"]
                if "Placeholder" in v["tags"]:
                    placeholder_positions.append(v["position"])
                    keys_to_delete.append(k)

            for k in keys_to_delete:
                del json_data[key]["objects"][k]

        elif key == "areas":
            for area_name in json_data[key]:
                keys_to_delete = []
                for k, v in json_data[key][area_name]["objects"].items():
                    if "Target" in v["tags"]:
                        v["color"] = target_color["rgb"]
                        target_positions[tuple(v["position"])] = v["color"]
                    if "Distractor" in v["tags"]:
                        v["color"] = np.random.choice(colorset)["code"]
                        distractor_positions[tuple(v["position"])] = v["color"]
                    if "Placeholder" in v["tags"]:
                        keys_to_delete.append(k)
                        placeholder_positions.append(v["position"])

                for k in keys_to_delete:
                    del json_data[key][area_name]["objects"][k]


    # round the positions to two decimals
    target_positions = {tuple([round(p, 2) for p in pos]): color for pos, color in target_positions.items()}
    distractor_positions = {tuple([round(p, 2) for p in pos]): color for pos, color in distractor_positions.items()}
    # placeholder_positions = [tuple([round(p, 2) for p in pos]) for pos in placeholder_positions]

    # add one empty line to the end of the xml file for the sake of the regex
    xml_data.append("\n")

    # modify the xml file based on the previously stored positions.
    # The xml file is a list of strings, each string representing a line in the xml file.

    for i, line in enumerate(xml_data.copy()):

        if str("pos=") in line:    
            xml_pos = line.split("pos=")[1].split('"')[1].split(" ")
            xml_pos = tuple([round(float(p), 2) for p in xml_pos])
            
            if xml_pos in target_positions.keys():
                # Match 'rgba="<any_value>"'
                pattern = r'rgba="[^"]*"'
                replacement = f'rgba="{target_positions[xml_pos]}"'
                xml_data[i+1] = re.sub(pattern, replacement, line)

            elif xml_pos in distractor_positions.keys():
                # Match 'rgba="<any_value>"'
                pattern = r'rgba="[^"]*"'
                replacement = f'rgba="{distractor_positions[xml_pos]}"'
                xml_data[i+1] = re.sub(pattern, replacement, line)


    with open(json_file_path, "w") as f:
        json.dump(json_data, f, indent=2)

    with open(xml_file_path, "w") as f:
        f.writelines(xml_data)


def write_environments(
    prompts_file_path,
    yml_output_dir_path,
    xml_output_dir_path,
    xml_object_dir_path,
    colorset_file_path,
    color_amount,
):
    """Generates xml files in xml_output_dir_path based on the entries in the json file

    Args:
        yml_output_dir_path (str): path to yml directory
        xml_output_dir_path (str): path to output directory

    Returns:
        none
    """
    with open(prompts_file_path, "r") as f:
        data = json.load(f)

    object_pool = get_object_pool(data)

    if len(data) == 0:
        raise ValueError("No prompts found in prompts file")

    for entry in data:
        write_yml_entry(
            entry=entry,
            yml_output_dir_path=yml_output_dir_path,
            object_pool=object_pool,
        )
        write_xml_entry(
            entry=entry,
            yml_output_dir_path=yml_output_dir_path,
            xml_output_dir_path=xml_output_dir_path,
            xml_object_dir_path=xml_object_dir_path,
            colorset_file_path=colorset_file_path,
            color_amount=color_amount,
        )


def main():
    # Define paths
    json_file = os.path.join("json_files", "prompts.json")
    yml_output_dir_path = os.path.join("yml_files")
    xml_output_dir_path = os.path.join("xml_files")
    xml_object_dir_path = os.path.join("objects")

    # Write ymls and xmls
    print("Using objects stored at", xml_object_dir_path)
    print("Writing ymls at", yml_output_dir_path)
    print("Writing xmls at", xml_output_dir_path)
    write_environments(
        json_file=json_file,
        yml_output_dir_path=yml_output_dir_path,
        xml_output_dir_path=xml_output_dir_path,
        xml_object_dir_path=xml_object_dir_path,
    )


if __name__ == "__main__":
    main()
