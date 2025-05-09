import json
from collections import Counter
import matplotlib.pyplot as plt

def compare_texture_counts(json_file_path1, json_file_path2):
    # Load the JSON data from both files
    with open(json_file_path1, 'r') as file1, open(json_file_path2, 'r') as file2:
        data1 = json.load(file1)
        data2 = json.load(file2)

    # Flatten the JSON values into single lists of textures
    textures1 = [texture for texture_list in data1.values() for texture in texture_list]
    textures2 = [
        texture for key, texture_list in data2.items()
        if "original" in key for texture in texture_list
    ]

    # Print the keys being used in textures2
    keys_used_in_textures2 = [key for key in data2.keys() if "original" in key]
    print("Keys used in textures2:", keys_used_in_textures2)

    # Count occurrences of each texture
    counts1 = Counter(textures1)
    counts2 = Counter(textures2)

    # Calculate total counts for normalization
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())

    # Combine all unique textures from both files
    all_textures = set(counts1.keys()).union(set(counts2.keys()))

    # Prepare data for the chart as normalized counts
    textures = list(all_textures)
    normalized_counts_file1 = [counts1.get(texture, 0) / total1 for texture in textures]
    normalized_counts_file2 = [counts2.get(texture, 0) / total2 for texture in textures]

    # Sort textures and normalized counts by normalized_counts_file1
    sorted_data = sorted(zip(textures, normalized_counts_file1, normalized_counts_file2), key=lambda x: x[1], reverse=True)
    textures, normalized_counts_file1, normalized_counts_file2 = zip(*sorted_data)

    # Plot the bar chart
    x = range(len(textures))
    width = 0.4

    plt.bar(x, normalized_counts_file1, width=width, label='Test Results', align='center')
    plt.bar([i + width for i in x], normalized_counts_file2, width=width, label='Training Data', align='center')

    plt.xlabel('Textures')
    plt.ylabel('Normalized Counts')
    plt.title('Texture Counts Comparison (Normalized)')
    plt.xticks([i + width / 2 for i in x], textures, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
json_file_path1 = '../test_matches_multilabel.json'
json_file_path2 = 'dominant_color_matches.json'
compare_texture_counts(json_file_path1, json_file_path2)