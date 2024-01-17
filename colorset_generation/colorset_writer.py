import json
import csv
import os


def filter_by_word_count(colors, max_words):
    """Returns list of all colors which have fewer than max_words in their name"""

    return [color for color in colors if len(color.split()) <= (max_words + 1)]


def hex_to_rgb(hex_color):
    """Returns the RGB tuple corresponding to a hex color code"""

    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def hex_to_rgba(hex_color):
    """Returns the RGBA tuple corresponding to a hex color code without alpha (sets alpha = 1.0))"""

    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)) + (1.0,)


def format_output(colors, output_format, color_format, max_words, output_path):
    """Stores colors in the selected format

    Args:
        colors (list): colors as stored in the dataset
        output_format (str): csv, txt or json for the file format
        color_format (str): hex or rgb for the color format

    Returns:
        none
    """
    formatted_colors = []
    for color in colors:
        name, code, newline = color.split("\t")
        if color_format == "hex":
            code = code.strip()
        elif color_format == "rgb":
            code = hex_to_rgb(code)
        elif color_format == "rgba":
            code = hex_to_rgba(code)

        formatted_colors.append({"name": name, "code": code})

    output_filename = f"colors_{max_words}words_{color_format}"
    output_path = os.path.join(output_path, output_filename)

    if output_format == "csv":
        with open(output_path + ".csv", "w", newline="") as csvfile:
            fieldnames = ["name", "code"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(formatted_colors)
    elif output_format == "txt":
        with open(output_path + ".txt", "w") as txtfile:
            for color in formatted_colors:
                txtfile.write(f"{color['name']} , {color['code']}\n")
    elif output_format == "json":
        with open(output_path + ".json", "w") as jsonfile:
            json.dump(formatted_colors, jsonfile, indent=2)


def generate_colorset(
    max_words, output_format, color_format, dataset_path, output_path
):
    """Generates a colorset with the given parameters

    Args:
        max_words (int): maximum number of words in a color name
        output_format (str): csv, txt or json for the file format
        color_format (str): hex or rgb for the color format

    Returns:
        none
    """
    with open(dataset_path, "r") as file:
        colors = file.readlines()[1:]  # Ignore the first line

    filtered_colors = filter_by_word_count(colors=colors, max_words=max_words)

    format_output(
        colors=filtered_colors,
        output_format=output_format,
        color_format=color_format,
        max_words=max_words,
        output_path=output_path,
    )


def main():
    max_words = int(input("Enter the maximum number of words for colors: "))
    output_format = input("Enter the output format (csv, txt, json): ")
    color_format = input("Enter the color format (hex, rgb, rgba): ")
    output_path = os.path.join(os.getcwd(), "output")

    generate_colorset(max_words, output_format, color_format, output_path)

    print(f"Output saved in {output_format} format in {output_path}")


if __name__ == "__main__":
    main()
