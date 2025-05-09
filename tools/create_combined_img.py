import os
import json
from PIL import Image

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

training_dir = "../testing_image_dataset"  # Directory containing training images
texture_dir = "../textures"  # Directory containing texture images
matches_file = "../test_matches_multilabel.json" # JSON file with matches
output_dir = "../combined_test_images"  # Output directory for combined images
create_combined_images(training_dir, texture_dir, matches_file, output_dir)