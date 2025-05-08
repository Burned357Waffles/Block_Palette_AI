import os
import json
from PIL import Image, ImageDraw

def display_block_with_avg_color(json_file, textures_dir, output_image, block_size=100):
    """
    Displays the average color of each block next to its image.

    :param json_file: Path to the JSON file containing average RGB values.
    :param textures_dir: Directory containing block images.
    :param output_image: Path to save the output image.
    :param block_size: Size of each block (width and height).
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"{json_file} not found!")
    if not os.path.exists(textures_dir):
        raise FileNotFoundError(f"{textures_dir} directory not found!")

    # Load the JSON data
    with open(json_file, "r") as f:
        rgb_data = json.load(f)

    # Prepare the output image
    num_blocks = len(rgb_data)
    img_width = block_size * 2
    img_height = block_size * num_blocks
    output_img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(output_img)

    for i, (block_name, rgb) in enumerate(rgb_data.items()):
        y_start = i * block_size
        y_end = y_start + block_size

        # Load the block image
        block_image_path = os.path.join(textures_dir, block_name)
        if not os.path.exists(block_image_path):
            print(f"Warning: Texture for {block_name} not found!")
            continue

        block_img = Image.open(block_image_path).resize((block_size, block_size))

        # Draw the block image
        output_img.paste(block_img, (0, y_start))

        # Draw the average color
        avg_color = (rgb["r"], rgb["g"], rgb["b"])
        draw.rectangle([block_size, y_start, block_size * 2, y_end], fill=avg_color)

    # Save the output image
    output_img.save(output_image)
    print(f"Composite image saved to {output_image}")


# Example usage
json_file = "average_block_rgb_values.json"
textures_dir = "../textures"
output_image = "composite_blocks.png"
display_block_with_avg_color(json_file, textures_dir, output_image)