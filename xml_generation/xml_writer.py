# Goal here is to build a yml, call PITA on it, then modify the resulting xml to fit my experiment. 
# This script should only generate a single set of xml files based on the json file it gets. 

# The yml should be as follows: 
# Agent in one half, five random objects in another half. 
#   - at least one object of the same type as the target object, but of different color OR at least one object of the same color as the target object, but of different type
#   - the color choices should be from the colorset pool of rgb values
# 
# Then, we will change the color of the objects according to how we need them to be:
#   - iterate over the objects. note: potentially shuffle them, to avoid biases
#   - the first one of target type is the target object:
#       - give it the correct color, and rename it TARGET
#   - all others get a random color such that the conditions are met (see above)

import os
import yaml
import json
    

def write_yaml_entry(f, entry, yaml_output_path):

    # Define the fixed structure for the YAML file
    yaml_data = {
        "Environment": {
            "size_range": [100, 100],
            "Style": [{"pretty_mode": False}],
            "Borders": [
                {"xml_name": "Border.xml"},
                {"place": True},
                {"tags": ["Border"]}
            ],
            "Objects": {}
        }
    }

    for obj_name, obj_data in entry.items():
        obj_structure = [
            {"xml_name": f"{obj_name}.xml"},
            {"amount": obj_data["amount"]},
            {"z_rotation_range": obj_data["z_rotation_range"]},
            {"tags": ["Agent"]}
        ]
        if "coordinates" in obj_data:
            obj_structure.append({"coordinates": obj_data["coordinates"]})
        if "color_groups" in obj_data:
            obj_structure.append({"color_groups": obj_data["color_groups"]})
        if "size_groups" in obj_data:
            obj_structure.append({"size_groups": obj_data["size_groups"]})
        if "size_value_range" in obj_data:
            obj_structure.append({"size_value_range": obj_data["size_value_range"]})

        yaml_data["Environment"]["Objects"][obj_name] = obj_structure

    # Add five specific objects
    for i in range(0, 5):
        if i == 0:
            #obj_name = "Target"
            if "shape" in entry and "xml_name" in entry["shape"]:
                obj_name = entry["shape"]["xml_name"]
            else:
                raise ValueError("No shape specified for target object")
            
            obj_structure = [
                {"xml_name": f"{obj_name}.xml"},
                {"z_rotation_range": [-180, 180]},
                {"tags": ["Target"]}
            ]
        else:
            #obj_name = f"Distractor{i}"
            obj_structure = [
                {"z_rotation_range": [-180, 180]},
                {"coordinates": [[i, i, i], [2*i, 2*i, 2*i]]}, # TODO put in coordinates that are in range for my env
                {"tags": ["Distractor"]}
            ]


        yaml_data["Environment"]["Objects"][obj_name] = obj_structure

    # Write YAML data to file
    entry_name = entry["prompt"].replace(" ", "_")
    yaml_output_path = os.path.join(yaml_output_path, f"{entry_name}.yml")
    
    with open(yaml_output_path, "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)



def write_yamls(json_file, yaml_output_path, data):

    with open(json_file, "r") as f:
        data = json.load(f)
    return data

    for entry in data:
        # rename the entry to be entry["name"] but with all whitepace replaced with underscore
        with open(yaml_file, "w") as f:
            write_yaml_entry(f, entry, yaml_output_path)

