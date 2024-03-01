import os
import shutil

from colorset_generation import colorset_writer
from prompt_generation import prompt_writer
from xml_generation import xml_writer

# TODO write test levels

def get_default_level_parameters(level_amount):
    """Return default level parameters as a dict of lists of ints

    Args:
        level_amount (int): total amount of levels in the curriculum

    Returns:
        dict: dict of lists of ints of length level_amount
    """
    color_amounts = [2, 2, 3, 3, 4, 5, 6, 6, 6][:level_amount]
    shape_amounts = [1, 2, 2, 3, 3, 4, 5, 6, 6][:level_amount]
    size_amounts = [1, 1, 2, 2, 2, 2, 2, 2, 2][:level_amount]
    instr_type_amounts = 2
    instr_amounts = [
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5][:level_amount],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1][:level_amount],
    ][:instr_type_amounts]

    return {
        "color_amounts": color_amounts,
        "shape_amounts": shape_amounts,
        "size_amounts": size_amounts,
        "instr_type_amounts": instr_type_amounts,
        "instr_amounts": instr_amounts,
    }


def get_custom_level_parameters():
    # TODO
    return None


def generate_level(
    level_number,
    colorset, # TODO rename colorset_file_path
    color_amount,
    shape_amount,
    size_amount,
    instr_amounts,
    curriculum_dir_path,
    instr_types,
    instr_file_path,
    xml_objects_file_path,
):
    print(f"Generating level {level_number}...")
    print("color_amount:", color_amount)
    print("shape_amount:", shape_amount)
    print("size_amount:", size_amount)
    print("instr_types:", instr_types)
    print("instr_amounts:", instr_amounts)

    level_folder = f"level_{level_number}"
    level_dir_path = os.path.join(curriculum_dir_path, level_folder)

    if os.path.exists(level_dir_path):
        shutil.rmtree(level_dir_path)
    os.mkdir(level_dir_path)  # TODO incorporate this logic into all relevant scripts

    # prompts file path is curriculum_dir_path/level_dir_path/prompts/prompts.json
    prompts_file_path = os.path.join(level_dir_path, "prompts", "prompts.json")
    # os.makedirs(os.path.dirname(prompts_file_path), exist_ok=True)

    # generate prompts. prompt path is curriculum_dir_path/level_dir_path/prompts/prompts.json
    prompt_writer.write_prompts(
        colorset=colorset,
        prompts_file_path=prompts_file_path,
        xml_objects_file_path=xml_objects_file_path,
        color_amount=color_amount,
        shape_amount=shape_amount,
        size_amount=size_amount,
        instr_amounts=instr_amounts,
    )

    # xml path for the level is curriculum_dir_path/level_dir_path/xml
    # yml path for the level is curriculum_dir_path/level_folder/yml
    xml_output_dir_path = os.path.join(level_dir_path, "xml")
    yml_output_dir_path = os.path.join(level_dir_path, "yml")

    # generate xmls
    xml_writer.write_environments(
        prompts_file_path=prompts_file_path,
        xml_output_dir_path=xml_output_dir_path,
        xml_objects_file_path=xml_objects_file_path,
        yml_output_dir_path=yml_output_dir_path,
        colorset_file_path=colorset_file_path,
        color_amount=color_amount,
    )


def generate_curriculum(curriculum_dir_path, level_amount, level_parameters):
    # iterate over level_parameters, which is a dict of lists of ints. The keys are the parameters for the generate_level function.
    # within this function, call generate_level with the parameters from the dict

    for (
        level_number,
        color_amount,
        shape_amount,
        size_amount,
        instr_amounts,
    ) in level_parameters:
        generate_level(
            level_number,
            color_amount,
            shape_amount,
            size_amount,
            instr_amounts,
            curriculum_dir_path,
        )


# Function that takes the generated levels of the highest level, and removes 1/3 of the content of each, then returns all the removed levels
def get_test_levels(levels):
    # TODO
    return None


def move_test_levels(test_levels, output_path):
    # TODO move test levels to different folder
    return None


def main():
    xkcd_dataset_file_path = os.path.join(os.getcwd(), "data", "color_data", "rgb.txt")
    colorset_dir_path = os.path.join(os.getcwd(), "data", "color_data")
    prompts_file_path = os.path.join(os.getcwd(), "data", "prompt_data", "prompts.json")
    curriculum_dir_path = os.path.join(
        os.getcwd(), "data", "curriculum"
    )  # TODO unify naming convention (dir vs file)
    xml_object_dir_path = os.path.join(os.getcwd(), "data", "objects")
    instr_file_path = os.path.join(os.getcwd(), "data", "prompt_data", "instructions.txt")
    instr_types = ["approach", "avoid"]
    size_modifiers = [
        {"name": "large", "factor": 2},
        {"name": "small", "factor": 0.5},
        {"name": "huge", "factor": 4},
        {"name": "tiny", "factor": 0.25}
    ]


    level_amount = 3
    params = get_default_level_parameters(level_amount)

    colorset_writer.generate_colorset(
        max_words=1,  # word amount per color (e.g. "green" vs "dark green")
        output_format="json",
        color_format="rgba",
        dataset_path=xkcd_dataset_file_path,
        output_dir_path=colorset_dir_path,
    )
    colorset_file_path = os.path.join(colorset_dir_path, "colors.json")

    # generate one set of prompts and their corresponding xmls for each level
    # the amount of levels
    for level_number in range(level_amount):
        # generate a new directory for each level
        level_dir_path = os.path.join(curriculum_dir_path, f"level_{level_number}")
        if os.path.exists(level_dir_path):
            shutil.rmtree(level_dir_path)
        os.mkdir(level_dir_path)

        prompts_file_path = os.path.join(
            level_dir_path, f"prompts_lvl{level_number}.json"
        )
        
        instr_amounts = [instr_amount[level_number] for instr_amount in params["instr_amounts"]]

        # generate prompts
        prompt_writer.write_prompts(
            colorset_file_path=colorset_file_path,
            output_file_path=prompts_file_path,
            xml_objects_dir_path=xml_object_dir_path,
            instr_file_path=instr_file_path,
            color_amount=params["color_amounts"][level_number],
            shape_amount=params["shape_amounts"][level_number],
            size_amount=params["size_amounts"][level_number],
            instr_amounts=instr_amounts,
            size_modifier_list=size_modifiers,
            instr_types=instr_types, # TODO move to params
        )

        # generate xmls
        xml_writer.write_environments(
            prompts_file_path=prompts_file_path,
            yml_output_dir_path=level_dir_path,
            xml_output_dir_path=level_dir_path,
            xml_object_dir_path=xml_object_dir_path,
            colorset_file_path=colorset_file_path,
            color_amount=params["color_amounts"][level_number],
            size_modifier_list=size_modifiers,
            size_amount=params["size_amounts"][level_number],
        )

    #generate_curriculum(curriculum_dir_path, params)



if __name__ == "__main__":
    main()
