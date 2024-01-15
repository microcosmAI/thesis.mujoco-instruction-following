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

# TODO decide on yml vs yaml

import os
import yaml
import json
import numpy as np
from pita_algorithm.pita import PITA


def write_yaml_entry(entry, yaml_output_path, object_pool):
    """Write a single yaml file based on one entry in the json file

    Args:
        entry (dict): Entry describing the target object and prompt
        yaml_output_path (str): Path to output directory
        object_pool (list): List of all object types that occur in the json file

    Raises:
        ValueError: If no object shape is found in the entry
    """
    # Define the fixed structure for the YAML file
    yaml_data = {
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
        yaml_data["Areas"]["Area2"]["Objects"][obj_name] = obj_structure

    # Get filename from prompt
    entry_name = entry["prompt"].replace(" ", "_").lower()
    yaml_output_path = os.path.join(yaml_output_path, f"{entry_name}.yml")

    os.makedirs(os.path.dirname(yaml_output_path), exist_ok=True)

    # with open(yaml_output_path, 'w') as yaml_file:
    #    yaml.dump(yaml_data, yaml_file, default_flow_style=None, indent=2)

    # Convert data to YAML
    yaml_str = yaml.dump(yaml_data, default_flow_style=None, indent=2)

    # Alter the YAML string formatting to the current expected format for PITA (subject to change)
    yaml_str = yaml_str.replace("{", "").replace("}", "").replace("'", "")

    # Write the modified YAML string to file
    with open(yaml_output_path, "w") as yaml_file:
        yaml_file.write(yaml_str)


def write_levels(json_file, yaml_output_path, xml_output_path, xml_object_path):
    """Generates xml files in xml_output_path based on the entries in the json file

    Args:
        yaml_output_path (str): path to yaml directory
        xml_output_path (str): path to output directory

    Returns:
        none
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    object_pool = get_object_pool(data)

    for entry in data:
        write_yaml_entry(
            entry=entry, yaml_output_path=yaml_output_path, object_pool=object_pool
        )
        write_xml_entry(
            entry=entry,
            yaml_output_path=yaml_output_path,
            xml_output_path=xml_output_path,
            xml_object_path=xml_object_path,
        )


def write_xml_entry(entry, yaml_output_path, xml_object_path, xml_output_path):
    """Generates a single xml file based on a single yaml file, as well as the details from the json file"""
    # get yaml/xml filenames from json entry
    yaml_path = os.path.join(
        yaml_output_path, entry["prompt"].replace(" ", "_").lower() + ".yml"
    )
    xml_path = os.path.join(
        xml_output_path, entry["prompt"].replace(" ", "_").lower())

    xml_object_path = os.path.join(xml_object_path)

    # call PITA
    PITA().run(
        random_seed=None,
        config_path=yaml_path,
        xml_dir=xml_object_path,
        export_path=xml_path,
        plot=False,
    )

    # TODO change the colors of the objects in the xml file according to my experiment


def write_xmls(yaml_output_path, xml_output_path, xml_object_path):
    """Generates xml files in xml_output_path based on the yamls in yaml_output_path

    Args:
        yaml_output_path (str): path to yaml directory
        xml_output_path (str): path to output directory

    Returns:
        none

    Should be obsolete, since we can just call PITA on each yaml directly
    """
    # iterate over all yamls in yaml_output_path, call PITA on each entry, and save the resulting xml in xml_output_path
    for filename in os.listdir(yaml_output_path):
        if filename.endswith(".yml"):
            yaml_path = os.path.join(yaml_output_path, filename)
            xml_path = os.path.join(xml_output_path, filename.split(".")[0] + ".xml")
            xml_object_path = os.path.join(xml_object_path)

            # call PITA
            PITA().run(
                random_seed=None,
                config_path=yaml_path,
                xml_dir=xml_object_path,
                export_path=xml_path,
                plot=False,
            )

            # TODO change the colors of the objects in the xml file according to my experiment
        else:
            continue


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
    yaml_output_path = os.path.join("yml_files")
    xml_output_path = os.path.join("xml_files")
    xml_object_path = os.path.join("xml_objects")

    # Write yamls and xmls
    print("Using objects stored at", xml_object_path)
    print("Writing yamls at", yaml_output_path)
    print("Writing xmls at", xml_output_path)
    write_levels(
        json_file=json_file,
        yaml_output_path=yaml_output_path,
        xml_output_path=xml_output_path,
        xml_object_path=xml_object_path,
    )


if __name__ == "__main__":
    main()
