from ultralytics import YOLO
import cv2
import numpy as np

# Load the fine-tuned YOLO model
model = YOLO('runs/detect/train/weights/best.pt')  # Update path to best.pt if needed

def predict(image):
    # Perform inference
    results = model.predict(source=image, save=False)
    detections = []  # To store detection details

    # Process results and draw bounding boxes on the image
    for result in results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            # Extract box coordinates, confidence score, and class label
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class label
            label = result.names[class_id]  # Class name

            # Append detection details
            detections.append({"class": label, "confidence": confidence, "bbox": (x1, y1, x2, y2)})

            # Draw bounding box and label on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, detections  # Return the image with detections and the detections list
