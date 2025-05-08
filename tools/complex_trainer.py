import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

# Path to the COCO-Stuff pretrained model checkpoint
MODEL_PATH = "models/deeplabv3_resnet101_coco-stuff.pth"

# Load DeepLabV3 model architecture (no pretrained weights)
from torchvision.models.segmentation import deeplabv3_resnet101
model = deeplabv3_resnet101(pretrained=False, num_classes=182)  # COCO-Stuff150 + background = ~182
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# Image transform
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# COCO-Stuff150 label index for "house"
HOUSE_INDEX = 128

def segment_house(image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    house_mask = (output_predictions == HOUSE_INDEX).astype(np.uint8) * 255
    cv2.imwrite(output_path, house_mask)

# Batch process
input_dir = "../training_image_dataset"
output_dir = "../combined_images/masks"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        segment_house(input_path, output_path)
