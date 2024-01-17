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


def write_yml_entry(entry, yml_output_path, object_pool):
    """Write a single yml file based on one entry in the json file

    Args:
        entry (dict): Entry describing the target object and prompt
        yml_output_path (str): Path to output directory
        object_pool (list): List of all object types that occur in the json file

    Raises:
        ValueError: If no object shape is found in the entry
    """
    # Define the fixed structure for the yml file
    yml_data = {
        "Environment": {
            "size_range": [1000, 1000],
            "Style": [{"pretty_mode": False}],
            "Borders": [
                {"xml_name": "Border.xml"},
                {"place": True},
                {"tags": ["Border"]},
            ],
            "Objects": {
                "AgentPlaceholder": [
                        {"xml_name": "BoxAgent.xml"},
                        {"amount": 1},
                        {"z_rotation_range": [-180, 180]},
                        {"coordinates": [[25, 50, 3]]}, # TODO test placement of agent
                        {"tags": ["Agent"]},
                ],
            },
        },
        "Areas": {
            "Area1": {
                "Objects": {
                    # TODO place agent here if random placement is needed
                    "placeholder_box": [
                        {"xml_name": "Box.xml"}, 
                        {"amount": [1, 1]},
                        {"z_rotation_range": [-180, 180]},
                        {"tags": ["Placeholder"]},
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
                {"amount": [1, 1]}, # Needs to be a range for PITA to generate a random position
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
        obj_name = f"{obj_name}{i}" # avoid duplicates
        yml_data["Areas"]["Area2"]["Objects"][obj_name] = obj_structure

    # Get filename from prompt
    entry_name = entry["prompt"].replace(" ", "_").lower()
    yml_output_path = os.path.join(yml_output_path, f"{entry_name}.yml")

    os.makedirs(os.path.dirname(yml_output_path), exist_ok=True)

    # with open(yml_output_path, 'w') as yml_file:
    #    yaml.dump(yml_data, yml_file, default_flow_style=None, indent=2)

    # Convert data to yml
    yml_str = yaml.dump(yml_data, default_flow_style=None, indent=2)

    # Alter the yml string formatting to the current expected format for PITA (subject to change)
    yml_str = yml_str.replace("{", "").replace("}", "").replace("'", "")

    # Write the modified yml string to file
    with open(yml_output_path, "w") as yml_file:
        yml_file.write(yml_str)


def write_environments(prompts_file_path, yml_output_path, xml_output_path, xml_object_path):
    """Generates xml files in xml_output_path based on the entries in the json file

    Args:
        yml_output_path (str): path to yml directory
        xml_output_path (str): path to output directory

    Returns:
        none
    """
    with open(prompts_file_path, "r") as f:
        data = json.load(f)

    object_pool = get_object_pool(data)

    for entry in data:
        write_yml_entry(
            entry=entry, yml_output_path=yml_output_path, object_pool=object_pool
        )
        write_xml_entry(
            entry=entry,
            yml_output_path=yml_output_path,
            xml_output_path=xml_output_path,
            xml_object_path=xml_object_path,
        )


def write_xml_entry(entry, yml_output_path, xml_object_path, xml_output_path):
    """Generates a single xml file based on a single yml file, as well as the details from the json file"""
    # get yml/xml filenames from json entry
    yml_path = os.path.join(
        yml_output_path, entry["prompt"].replace(" ", "_").lower() + ".yml"
    )
    xml_path = os.path.join(
        xml_output_path, entry["prompt"].replace(" ", "_").lower())

    xml_object_path = os.path.join(xml_object_path)

    # call PITA
    PITA().run(
        random_seed=None,
        config_path=yml_path,
        xml_dir=xml_object_path,
        export_path=xml_path,
        plot=False,
    )

    # TODO change the colors of the objects in the xml file according to my experiment


def get_object_pool(data):
    """Returns a list of every object type (=xml_name) that occurs in data"""

    object_pool = []
    for entry in data:
        if "shape" in entry and "xml_name" in entry["shape"]:
            xml_name = entry["shape"]["xml_name"]
            if xml_name not in object_pool:
                object_pool.append(entry["shape"]["xml_name"])
        else:
            raise ValueError("No shape specified for target object")

    return object_pool


def main():
    # Define paths
    json_file = os.path.join("json_files", "prompts.json")
    yml_output_path = os.path.join("yml_files")
    xml_output_path = os.path.join("xml_files")
    xml_object_path = os.path.join("xml_objects")

    # Write ymls and xmls
    print("Using objects stored at", xml_object_path)
    print("Writing ymls at", yml_output_path)
    print("Writing xmls at", xml_output_path)
    write_environments(
        json_file=json_file,
        yml_output_path=yml_output_path,
        xml_output_path=xml_output_path,
        xml_object_path=xml_object_path,
    )


if __name__ == "__main__":
    main()
