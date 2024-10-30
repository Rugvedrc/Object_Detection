import torch
import cv2
import numpy as np
from albumentations import Compose, Resize, Normalize
import sys
import os
from PIL import Image
from io import BytesIO

# Append the YOLOv5 path to the system path
sys.path.append('E:/yolov5-master')

from models.common import DetectMultiBackend

# Load your fine-tuned YOLO model
def load_model(model_path):
    model = DetectMultiBackend(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

# Define the preprocessing function using Albumentations
def preprocess_image(image_file):
    # Open the image directly
    image = Image.open(image_file).convert("RGB")
    
    # Convert to numpy array and preprocess
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    transform = Compose([
        Resize(height=640, width=640),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    
    augmented = transform(image=image)
    image = augmented['image']
    image = torch.tensor(image).float().permute(2, 0, 1) / 255.0
    return image.unsqueeze(0)

# Perform inference using the model
def perform_inference(model, preprocessed_image):
    with torch.no_grad():
        results = model(preprocessed_image)
    return results

# Process detection results
def process_detections(detections):
    processed_detections = []
    for detection in detections:
        # Convert to numpy array if detection is a tensor
        if isinstance(detection, torch.Tensor):
            detection_np = detection.cpu().numpy()
        else:
            detection_np = detection  # If already a list or array, keep as is

        # Iterate over each item in the detection list or array
        for det in detection_np:
            # If `det` is still a list, convert it to a numpy array
            det = np.array(det) if isinstance(det, list) else det

            # Flatten the numpy array if possible
            det_flat = det.flatten() if hasattr(det, "flatten") else det

            # Ensure there are at least six values for the bounding box, confidence, and class
            if len(det_flat) >= 6:
                try:
                    x1, y1, x2, y2, conf, cls = map(float, det_flat[:6])
                    processed_detections.append((x1, y1, x2, y2, conf, cls))
                except (ValueError, IndexError) as e:
                    print(f"Error processing detection: {e}")
            else:
                print(f"Unexpected detection format: {det_flat}")
    return processed_detections

# Example usage
if __name__ == "__main__":
    model_path = "runs/detect/train/weights/best.pt"
    image_path = "JPEGImages/BloodImage_00007.jpg"
    
    model = load_model(model_path)
    preprocessed_image = preprocess_image(image_path)
    results = perform_inference(model, preprocessed_image)
    
    if isinstance(results, list) and len(results) > 0:
        process_detections(results)
    else:
        print("No detections found or unexpected results format:", type(results))
