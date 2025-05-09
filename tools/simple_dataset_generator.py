import os
import json
import math
import time
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms
import numpy as np


def get_representative_colors(image_path, num_colors=6, extra_clusters=2, scale_factor=256):
    """Extract num_colors that most represent the image using K-Means clustering."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # Ensure the image is in RGB mode
        img = img.resize((scale_factor, scale_factor), Image.Resampling.LANCZOS)
        pixels = np.array(img.getdata())  # Convert image data to a NumPy array

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=num_colors + extra_clusters, random_state=42)
        kmeans.fit(pixels)
        representative_colors = kmeans.cluster_centers_.astype(int)  # Get cluster centers as representative colors

        return [tuple(color) for color in representative_colors]

def load_json_data(json_file):
    """Load RGB data from a JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def is_blue_or_green_or_black(rgb, blue_threshold=1.142, green_threshold=1.04):
    """Check if the color is predominantly blue or green with a relaxed threshold."""
    r, g, b = rgb["r"], rgb["g"], rgb["b"]
    is_blue = b > r * (blue_threshold + 0.3) and b > g * blue_threshold
    is_green = g > r * green_threshold and g > b * (green_threshold + 0.25)
    is_black = r + g + b < 40 and max(r, g, b) < 15

    return is_blue or is_green or is_black

def find_closest_color(target_color, rgb_data, used_colors):
    """Find the closest color in the JSON data to the target color, avoiding duplicates and ignoring blue/green textures."""
    target_rgb = {"r": target_color[0], "g": target_color[1], "b": target_color[2]}
    if is_blue_or_green_or_black(target_rgb):
        #print(f"Skipping blue/green/black color: {target_color}")
        return None  # Skip this target color

    closest_color = None
    min_distance = float('inf')
    for block_name, rgb in rgb_data.items():
        if block_name in used_colors:
            continue  # Skip already used colors
        distance = math.sqrt(
            (target_color[0] - rgb["r"]) ** 2 +
            (target_color[1] - rgb["g"]) ** 2 +
            (target_color[2] - rgb["b"]) ** 2
        )
        if distance < min_distance:
            min_distance = distance
            closest_color = block_name
    if closest_color:
        used_colors.add(closest_color)

    return closest_color

def process_images(directory, json_file, num_colors=6, extra_clusters=2):
    """Process all images in the directory and find the closest matches for dominant colors."""
    rgb_data = load_json_data(json_file)
    results = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(directory, filename)
            representative_colors = get_representative_colors(image_path, num_colors, extra_clusters=extra_clusters)
            used_colors = set()  # Track used colors for each image
            closest_matches = []

            while len(closest_matches) < num_colors:
                #print(f"Finding closest matches for '{filename}'...")
                for color in representative_colors:
                    if len(closest_matches) >= num_colors:
                        break  # Stop once we have enough valid matches
                    match = find_closest_color(color, rgb_data, used_colors)

                    if match == "diorite.png" and "white_concrete.png" in closest_matches:
                        #print(f"Skipping 'diorite.png' for '{filename}' as 'white_concrete.png' is already matched.")
                        continue
                    elif match == "white_concrete.png" and "diorite.png" in closest_matches:
                        #print(f"Skipping 'white_concrete.png' for '{filename}' as 'diorite.png' is already matched.")
                        continue
                    if match:
                        closest_matches.append(match)

            results[filename] = closest_matches
    return results

def save_results_to_json(data, output_file):
    """Save the results to a JSON file."""
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def create_combined_images(training_dir, texture_dir, matches_file, output_dir):
    """Create combined images with training images and their matching textures placed horizontally below."""
    # Load the matches from the JSON file
    with open(matches_file, 'r') as f:
        matches = json.load(f)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for training_image, matching_textures in matches.items():
        training_image_path = os.path.join(training_dir, training_image)
        if not os.path.exists(training_image_path):
            print(f"Training image '{training_image}' not found. Skipping.")
            continue

        # Open the training image
        try:
            with Image.open(training_image_path) as img:
                training_img = img.copy()  # Use .copy() to avoid issues with closed files
        except Exception as e:
            print(f"Failed to open training image '{training_image}': {e}")
            continue

        # Open and resize each matching texture
        textures = []
        for texture in matching_textures:
            texture_path = os.path.join(texture_dir, texture)
            if os.path.exists(texture_path):
                try:
                    with Image.open(texture_path) as tex_img:
                        # Resize the texture image
                        scale_factor = 20
                        new_size = (tex_img.width * scale_factor, tex_img.height * scale_factor)
                        resized_tex_img = tex_img.resize(new_size, Image.NEAREST)
                        textures.append(resized_tex_img)
                except Exception as e:
                    print(f"Failed to open texture '{texture}': {e}")
            else:
                print(f"Texture '{texture}' not found. Skipping.")

        # Combine textures horizontally
        if textures:
            texture_widths, texture_heights = zip(*(tex.size for tex in textures))
            total_texture_width = sum(texture_widths)
            max_texture_height = max(texture_heights)

            combined_textures = Image.new("RGB", (total_texture_width, max_texture_height))
            x_offset = 0
            for tex in textures:
                combined_textures.paste(tex, (x_offset, 0))
                x_offset += tex.width

            scale_factor = combined_textures.width / training_img.width
            new_size = (combined_textures.width, int(training_img.height * scale_factor))
            training_img = training_img.resize(new_size, Image.NEAREST)

            # Combine training image and textures vertically
            total_width = combined_textures.width
            total_height = training_img.height + combined_textures.height


            combined_image = Image.new("RGB", (total_width, total_height))
            combined_image.paste(training_img, (0, 0))
            combined_image.paste(combined_textures, (0, training_img.height))

            # Save the combined image
            output_path = os.path.join(output_dir, f"combined_{training_image}")
            combined_image.save(output_path)
            #print(f"Saved combined image to '{output_path}'.")


def augment_images(input_dir, output_dir, num_augmentations=5):
    """
    Apply data augmentation to images in the input directory and save them to the output directory.

    Args:
        input_dir (str): Path to the directory containing original images.
        output_dir (str): Path to the directory to save augmented images.
        num_augmentations (int): Number of augmented versions to generate per image.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define augmentation transformations
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.3, hue=0.075),
        transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
    ])

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            with Image.open(image_path) as img:
                img = img.convert("RGB")  # Ensure the image is in RGB mode
                # save original image to output_dir
                original_filename = f"{os.path.splitext(filename)[0]}_original.jpg"
                img.save(os.path.join(output_dir, original_filename))

                # Generate augmented images
                for i in range(num_augmentations):
                    augmented_img = augmentation_transforms(img)
                    augmented_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
                    augmented_img.save(os.path.join(output_dir, augmented_filename))


if __name__ == "__main__":
    training_dir = "../training_image_dataset"  # Directory containing training images
    augmented_dir = "../augmented_images"  # Directory to save augmented images
    texture_dir = "../textures"  # Directory containing texture images
    json_file = "average_block_rgb_values.json"  # JSON file with RGB data
    output_file = "dominant_color_matches.json"  # Output JSON file
    matches_file = "dominant_color_matches.json"  # JSON file with matches
    output_dir = "../combined_images"  # Directory to save combined images

    if not os.path.exists(training_dir):
        print(f"Directory '{training_dir}' does not exist.")
    elif not os.path.exists(json_file):
        print(f"JSON file '{json_file}' does not exist.")
    elif not os.path.exists(texture_dir):
        print(f"Directory '{texture_dir}' does not exist.")
    elif not os.path.exists(augmented_dir):
        print(f"Directory '{augmented_dir}' does not exist.")
    elif not os.path.exists(output_dir):
        print(f"Directory '{output_dir}' does not exist.")
    else:
        print("Augmenting images...")
        augment_images(training_dir, augmented_dir, num_augmentations=5)

        #begin timer
        print("Processing images...")
        start_p = time.time()
        results = process_images(augmented_dir, json_file)
        print(f"Processed {len(results)} images.")
        save_results_to_json(results, output_file)
        end_p = time.time()
        print(f"Dominant color matches saved to '{output_file}'.")

        start_c = time.time()
        create_combined_images(augmented_dir, texture_dir, matches_file, output_dir)
        end_c = time.time()
        print(f"Combined images saved to '{output_dir}'.")

        print(f"Processing time: {end_p - start_p:.2f} seconds")
        print(f"Combining time: {end_c - start_c:.2f} seconds")
        print(f"Total time: {end_c - start_p:.2f} seconds")
