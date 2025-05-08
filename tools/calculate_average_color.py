import os
import json
import math
from PIL import Image
from itertools import combinations

block_names = [
    # concrete
    "white_concrete", "light_gray_concrete", "gray_concrete", "black_concrete",
    "red_concrete", "orange_concrete", "yellow_concrete", "lime_concrete",
    "green_concrete", "cyan_concrete", "light_blue_concrete", "blue_concrete",
    "purple_concrete", "magenta_concrete", "pink_concrete", "brown_concrete",

    # terracotta
    "white_terracotta", "light_gray_terracotta", "gray_terracotta", "black_terracotta",
    "red_terracotta", "orange_terracotta", "yellow_terracotta", "lime_terracotta",
    "green_terracotta", "cyan_terracotta", "light_blue_terracotta", "blue_terracotta",
    "purple_terracotta", "magenta_terracotta", "pink_terracotta", "brown_terracotta",
    "terracotta",

    # planks
    "oak_planks", "spruce_planks", "birch_planks", "jungle_planks",
    "acacia_planks", "dark_oak_planks", "crimson_planks", "warped_planks",
    "cherry_planks", "bamboo_planks", "mangrove_planks", "pale_oak_planks",

    # logs
    "oak_log", "spruce_log", "birch_log", "jungle_log", "acacia_log", "dark_oak_log",
    "cherry_log", "bamboo_block", "mangrove_log", "pale_oak_log",

    # stone types
    "stone", "smooth_stone", "andesite", "diorite", "granite",
    "stone_bricks", "mossy_stone_bricks", "blackstone",
    "cobbled_deepslate", "mossy_cobblestone", "end_stone",
    "tuff", "tuff_bricks", "calcite",

    # copper
    "copper_block", "exposed_copper", "weathered_copper", "oxidized_copper",

    # misc
    # quartz_block_bottom is smooth quartz
    # sandstone_top is smooth sandstone
    "bricks", "nether_bricks", "sandstone", "sandstone_top", "quartz_block_bottom",
    "honeycomb_block", "mud_bricks", "amethyst_block",
    "dark_prismarine", "prismarine_bricks", "dirt"
]

def verify_block_names(texture_dir):
    # Verify that all block names are valid and exist in the directory. If an image exitst that does not match a block name, it will be deleted from the folder

    # Directory containing the images
    # List of valid block names
    valid_block_names = block_names
    # List to store invalid block names
    invalid_block_names = []
    # Iterate through each file in the directory
    for filename in os.listdir(texture_dir):
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Check if the file name (without extension) is in the valid block names
            block_name = os.path.splitext(filename)[0]
            if block_name not in valid_block_names:
                invalid_block_names.append(block_name)
                # Delete the invalid image file
                os.remove(os.path.join(texture_dir, filename))
                print(f"Deleted invalid image: {filename}")
            else:
                print(f"Valid image: {filename}")
    # Print the list of invalid block names
    if invalid_block_names:
        print("Invalid block names found:")
        for invalid_block in invalid_block_names:
            print(f"- {invalid_block}")
    else:
        print("All block names are valid.")


def calculate_average_rgb(image_path):
    """Calculate the average RGB value of an image."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # Ensure the image is in RGB mode
        pixels = list(img.getdata())
        num_pixels = len(pixels)
        avg_r = sum(pixel[0] for pixel in pixels) // num_pixels
        avg_g = sum(pixel[1] for pixel in pixels) // num_pixels
        avg_b = sum(pixel[2] for pixel in pixels) // num_pixels
        return avg_r, avg_g, avg_b

def process_textures(directory):
    """Process all images in the directory and calculate their average RGB values."""
    results = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(directory, filename)
            avg_rgb = calculate_average_rgb(image_path)
            results[filename] = {"r": avg_rgb[0], "g": avg_rgb[1], "b": avg_rgb[2]}
    return results

def save_to_json(data, output_file):
    """Save the results to a JSON file."""
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    texture_dir = "../textures"  # Directory containing the images
    output_file = "average_block_rgb_values.json"  # Output JSON file

    print(len(block_names))

    if not os.path.exists(texture_dir):
        print(f"Directory '{texture_dir}' does not exist.")
    else:
        verify_block_names(texture_dir )
        average_rgb_values = process_textures(texture_dir)
        save_to_json(average_rgb_values, output_file)
        print(f"Average RGB values saved to '{output_file}'.")