import json

# Load the texture data
with open('average_block_rgb_values.json', 'r') as rgb_file:
    all_textures = set(json.load(rgb_file).keys())

# Load the matched textures
with open('dominant_color_matches.json', 'r') as matches_file:
    matched_textures = set()
    matches = json.load(matches_file)
    for textures in matches.values():
        matched_textures.update(textures)

# Find unmatched textures
unmatched_textures = all_textures - matched_textures

print("Unmatched textures:")
unmatched_textures = sorted(unmatched_textures)
for texture in unmatched_textures:
    print(texture)