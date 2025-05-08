import os
import json
import math
from PIL import Image, ImageDraw, ImageFont
from itertools import combinations

def delta_e(rgb1, rgb2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))

def draw_low_delta_e_pairs(pairs, output_image, width=100, height=50):
    """
    Draws color pairs with low delta E values next to each other in an image, with labels.

    :param pairs: List of tuples (name1, name2, delta_e, rgb1, rgb2)
    :param output_image: Path to save the output image
    :param width: Width of each color block
    :param height: Height of each color block
    """
    image_height = len(pairs) * height
    image_width = width * 2
    img = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(img)

    # Optional: Use a custom font (requires a .ttf file)
    # font = ImageFont.truetype("arial.ttf", 12)
    font = ImageFont.load_default()

    for i, (name1, name2, d_e, rgb1, rgb2) in enumerate(pairs):
        y_start = i * height
        y_end = y_start + height

        # Draw the first color block
        draw.rectangle([0, y_start, width, y_end], fill=rgb1)
        draw.text((5, y_start + 5), name1, fill="black", font=font)

        # Draw the second color block
        draw.rectangle([width, y_start, width * 2, y_end], fill=rgb2)
        draw.text((width + 5, y_start + 5), name2, fill="black", font=font)

    img.save(output_image)
    print(f"Image saved to {output_image}")

# Main logic
output_file = "average_block_rgb_values.json"

if not os.path.exists(output_file):
    raise FileNotFoundError(f"{output_file} not found!")

with open(output_file, "r") as f:
    rgb_data = json.load(f)

all_pairs = []
for (name1, rgb1), (name2, rgb2) in combinations(rgb_data.items(), 2):
    try:
        color1 = (rgb1['r'], rgb1['g'], rgb1['b'])
        color2 = (rgb2['r'], rgb2['g'], rgb2['b'])
        d_e = delta_e(color1, color2)
        all_pairs.append((name1, name2, d_e, color1, color2))
    except KeyError as e:
        print(f"Missing key in color data: {e}")

all_pairs.sort(key=lambda x: x[2])
top_pairs = all_pairs[:10]

for name1, name2, d_e, _, _ in top_pairs:
    print(f"{name1} ↔ {name2}: ΔE = {d_e:.2f}")

output_image = "low_delta_e_pairs.png"
draw_low_delta_e_pairs(top_pairs, output_image)