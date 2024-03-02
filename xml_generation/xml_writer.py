import os
import yaml
import json
import numpy as np
from pita_algorithm.pita import PITA
import re
import random


def write_yml_entry(entry, yml_output_dir_path, object_pool):
    """Write a single yml file based on one entry in the json file

    Args:
        entry (dict): entry describing the target object and prompt
        yml_output_dir_path (str): path to output directory
        object_pool (list): list of all object types that occur in the json file

    Raises:
        ValueError: if no object shape is found in the entry
    """

    # Define the fixed structure for the yml file
    yml_data = {
        # Environment size is fixed at 100x100
        "Environment": {
            "size_range": [16, 16],
            "Style": [{"pretty_mode": False}],
            "Borders": [
                {"xml_name": "Border.xml"},
                {"place": True},
                {"tags": ["Border"]},
                {"color_groups": [1, 4]},
            ],
            "Objects": {
                "Light": [
                    {"xml_name": "Light.xml"},
                    {"amount": 1},
                    {"coordinates": [[50, 50, 15]]},
                    {"tags": ["Light"]},
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
                        {"z_rotation_range": [270, 271]},
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

    yml_str = yaml.dump(yml_data, default_flow_style=None, indent=2)

    # Alter the yml string formatting to the current expected format for PITA (subject to change)
    yml_str = yml_str.replace("{", "").replace("}", "").replace("'", "")

    with open(yml_output_file_path, "w") as yml_file:
        yml_file.write(yml_str)


def write_xml_entry(
    entry,
    yml_output_dir_path,
    xml_object_dir_path,
    xml_output_dir_path,
    colorset_file_path,
    color_amount,
    size_modifier_list,
    size_amount,
):
    """
    Generates a single xml file based on a single yml file, using details from the json file containing the prompts

    Args:
        entry (dict): entry from the json file containing details for generating the xml file
        yml_output_dir_path (str): directory path where the yml output files are stored
        xml_object_dir_path (str): directory path where the xml object files are stored
        xml_output_dir_path (str): directory path where the xml output files will be stored
        colorset_file_path (str): file path of the colorset file
        color_amount (int): amount of colors the objects can have
        size_modifier_list (list): list (of dicts) of size modifiers for the objects
        size_amount (int): amount of sizes to be used from the size_modifier_list
    """

    # get yml/xml filenames from json entry
    yml_file_path = os.path.join(
        yml_output_dir_path, entry["prompt"].replace(" ", "_").lower() + ".yml"
    )
    xml_file_path = os.path.join(
        xml_output_dir_path, entry["prompt"].replace(" ", "_").lower() + ".xml"
    )

    xml_object_dir_path = os.path.join(xml_object_dir_path)

    # call PITA
    print(f'Writing xml for prompt "{entry["prompt"]}"...')
    PITA().run(
        random_seed=None,
        config_path=yml_file_path,
        xml_dir=xml_object_dir_path,
        export_path=xml_file_path.removesuffix(".xml"),
        plot=False,
    )

    # change the colors and sizes of the objects in the xml file
    modify_xml(
        xml_file_path,
        entry,
        colorset_file_path,
        color_amount,
        size_modifier_list,
        size_amount,
    )


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


def modify_xml(
    xml_file_path,
    entry,
    colorset_file_path,
    color_amount,
    size_modifier_list,
    size_amount,
):
    """
    Modifies the xml file at xml_file_path based on the entry in the json file.

    The modifications are as follows:
        - the target object gets the target color
        - the distractor objects get a random color from the colorset
        - the placeholder objects are removed

    This function replaces functionality that PITA doesn't have yet.
    Should PITA be able to assign colors to objects, this function can be replaced.

    Args:
        xml_file_path (str): path to xml file
        entry (dict): entry from json file
        colorset_file_path (str): path to colorset file
        color_amount (int): amount of colors to use from colorset file
    """

    # store target parameters (entry represents the prompt, which contains the target object parameters)
    target_color = entry["color"]
    target_class = entry["shape"]["xml_name"].split(".")[0]
    if "size" in entry:
        hasSize = True
        sizes = size_modifier_list[:size_amount]
        target_size = entry["size"]  # format: {"name": "large", "factor": 2}
        target_size_factor = 1
        if target_size:
            target_size_factor = target_size["factor"]
        target_size_dimensions = []  # concrete values for the size

    json_file_path = xml_file_path.replace(".xml", ".json")

    # get colorset, without target color and with right amount of colors
    with open(colorset_file_path, "r") as f:
        colorset = json.load(f)

    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    with open(xml_file_path, "r") as f:
        xml_data = f.readlines()

    colorset = colorset[:color_amount]
    target_positions = {}  # format: {position: color} (position is a tuple)
    distractor_positions = {}
    placeholder_positions = []

    # modify the json file, store positions (as object ID) for modifying the xml file
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
                    if hasSize:
                        v["size"] = [x * target_size_factor for x in v["size"]]
                        target_size_dimensions = v["size"]

                if "Distractor" in v["tags"]:
                    # Assign random values, until they are different from the target values
                    while True:
                        v["color"] = np.random.choice(colorset)["code"]
                        if hasSize:
                            new_size = random.choice(sizes)
                            v["size"] = [x * new_size["factor"] for x in v["size"]]
                        if (v["color"], v["class"], v["size"]) != (
                            target_color["rgb"],
                            target_class,
                            target_size_dimensions,
                        ):
                            break

                distractor_positions[tuple(v["position"])] = v

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
                        if hasSize:
                            v["size"] = [x * target_size_factor for x in v["size"]]
                            target_size_dimensions = v["size"]

                    if "Distractor" in v["tags"]:
                        # Assign random values, until they are different from the target values
                        while True:
                            v["color"] = np.random.choice(colorset)["code"]
                            if hasSize:
                                new_size = random.choice(sizes)
                                v["size"] = [x * new_size["factor"] for x in v["size"]]
                            if (v["color"], v["class"], v["size"]) != (
                                target_color["rgb"],
                                target_class,
                                target_size_dimensions,
                            ):
                                break

                    distractor_positions[tuple(v["position"])] = v

                    if "Placeholder" in v["tags"]:
                        keys_to_delete.append(k)
                        placeholder_positions.append(v["position"])

                for k in keys_to_delete:
                    del json_data[key][area_name]["objects"][k]

    # round the positions to two decimals for robust matching
    target_positions = {
        tuple([round(p, 2) for p in pos]): color
        for pos, color in target_positions.items()
    }
    distractor_positions = {
        tuple([round(p, 2) for p in pos]): color
        for pos, color in distractor_positions.items()
    }

    # needed for the regex
    xml_data.append("\n")

    # for updating agent position (required for the PITA version used, can be removed in future versions)
    agent_line = None

    # modify the xml file based on the previously stored positions and values
    for i, line in enumerate(xml_data.copy()):
        if str("pos=") in line:
            xml_pos = line.split("pos=")[1].split('"')[1].split(" ")
            xml_pos = tuple([round(float(p), 2) for p in xml_pos])

            if xml_pos in target_positions.keys():
                # Match 'rgba="<any_value>"'
                pattern = r'rgba="[^"]*"'
                replacement = f'rgba="{target_positions[xml_pos]}"'
                replacement = (
                    replacement.replace("[", "").replace("]", "").replace(",", "")
                )
                xml_data[i + 1] = re.sub(pattern, replacement, xml_data[i + 1])

                # Match 'size="<any_value>"'
                pattern = r'size="[^"]*"'
                replacement = f'size="{target_size_dimensions}"'
                replacement = (
                    replacement.replace("[", "").replace("]", "").replace(",", "")
                )
                xml_data[i + 1] = re.sub(pattern, replacement, xml_data[i + 1])

            elif xml_pos in distractor_positions.keys():
                # Match 'rgba="<any_value>"'
                pattern = r'rgba="[^"]*"'
                replacement = f'rgba="{distractor_positions[xml_pos]["color"]}"'
                replacement = (
                    replacement.replace("[", "").replace("]", "").replace(",", "")
                )
                xml_data[i + 1] = re.sub(pattern, replacement, xml_data[i + 1])

                # Match 'size="<any_value>"'
                pattern = r'size="[^"]*"'
                replacement = f'size="{distractor_positions[xml_pos]["size"]}"'
                replacement = (
                    replacement.replace("[", "").replace("]", "").replace(",", "")
                )
                xml_data[i + 1] = re.sub(pattern, replacement, xml_data[i + 1])

        # The agent position gets moved up the hierarchy in the xml file here, such that the freeJoint
        # and the agent geom are in the same parent tag. The child tags need to be positioned 0. 0. 0.
        # This is required to keep rotation and position intact within mujoco. 
        # This can be removed when using future versions of PITA, which will generate the xml files 
        # in the correct format.
        if 'name="agent/"' in line:
            agent_line = i

        if agent_line is not None and "pos=" in line:
            # Get the position and add it to the agent tag
            agent_pos = line.split('pos="')[1].split('"')[0]
            agent_pos = f'pos="{agent_pos}"'
            xml_data[agent_line] = xml_data[agent_line].replace(">", f" {agent_pos}>")

            # Set the position of the agent geom to 0 0 0
            pattern = r'pos="[^"]*"'
            replacement = 'pos="0 0 0"'
            xml_data[i] = re.sub(pattern, replacement, xml_data[i])

            agent_line = None

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
    size_modifier_list,
    size_amount,
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
            size_modifier_list=size_modifier_list,
            size_amount=size_amount,
        )
