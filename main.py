
import os
import json
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

IMAGE_SIZE = 256  # Image size for loading

# Load matches from JSON
def load_matches(json_file, image_dir, image_size=(IMAGE_SIZE, IMAGE_SIZE)):
    with open(json_file, 'r') as f:
        matches = json.load(f)

    images = []
    labels = []
    texture_names = list(set(texture for textures in matches.values() for texture in textures))
    texture_to_label = {texture: i for i, texture in enumerate(texture_names)}
    label_to_texture = {i: texture for texture, i in texture_to_label.items()}

    for image_name, textures in matches.items():
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            img = load_img(image_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0
            label_vector = np.zeros(len(texture_names))
            for texture in textures:
                if texture in texture_to_label:
                    label_vector[texture_to_label[texture]] = 1.0
            images.append(img_array)
            labels.append(label_vector)

    return np.array(images), np.array(labels), texture_to_label, label_to_texture

# Load data
image_dir = "augmented_images"
json_file = "tools/dominant_color_matches.json"
X, y, texture_to_label, label_to_texture = load_matches(json_file, image_dir)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(texture_to_label), activation='sigmoid')  # For multi-label output
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model
model.save("texture_classification_model_multilabel.h5")

# Predict on new data
testing_dir = "testing_image_dataset"
output_file = "test_matches_multilabel.json"

print("Processing images for testing...")
start_time = time.time()

test_images = []
test_filenames = []
for image_name in os.listdir(testing_dir):
    image_path = os.path.join(testing_dir, image_name)
    if os.path.exists(image_path):
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img) / 255.0
        test_images.append(img_array)
        test_filenames.append(image_name)

test_images = np.array(test_images)
predictions = model.predict(test_images)

# Get top 6 matches
results = {}
for i, image_name in enumerate(test_filenames):
    top_indices = predictions[i].argsort()[-6:][::-1]
    top_textures = [label_to_texture[idx] for idx in top_indices]
    results[image_name] = top_textures

# Save results
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print("Processing complete.")
print(f"Test matches saved to '{output_file}'.")
print(f"Processing time: {time.time() - start_time:.2f} seconds")
