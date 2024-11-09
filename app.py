import streamlit as st
import cv2
import numpy as np
import tempfile
import time

def load_yolo_model(config_path, weights_path, classes_path):
    # Load class names
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Generate random colors for each class
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    # Load YOLO model
    net = cv2.dnn.readNet(weights_path, config_path)
    
    return net, classes, colors

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def detect_objects(image, net, classes, colors):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    
    # Initialize lists for detected objects
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # Draw predictions
    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        
        x = round(box[0])
        y = round(box[1])
        w = round(box[2])
        h = round(box[3])
        
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        
        # Draw rectangle and label
        cv2.rectangle(image, (x,y), (x+w,y+h), color, 2)
        cv2.putText(image, f"{label} {confidences[i]:.2f}", (x-10,y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def main():
    st.title("YOLO Object Detection App")
    
    # Sidebar for model configuration
    st.sidebar.title("Configuration")
    config_path = st.sidebar.text_input("Config Path", "yolov3.cfg")
    weights_path = st.sidebar.text_input("Weights Path", "yolov3.weights")
    classes_path = st.sidebar.text_input("Classes Path", "yolov3.txt")
    
    # Load YOLO model
    try:
        net, classes, colors = load_yolo_model(config_path, weights_path, classes_path)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return
    
    # Mode selection
    mode = st.radio("Select Mode", ["Image Upload", "Webcam"])
    
    if mode == "Image Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Save uploaded file to temp location
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            # Read and process image
            image = cv2.imread(tfile.name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Perform detection
            processed_image = detect_objects(image.copy(), net, classes, colors)
            
            # Display results
            st.image(processed_image, caption='Processed Image',  use_container_width=True)
    
    else:  # Webcam mode
        st.write("Webcam Object Detection")
        run = st.checkbox('Start/Stop')
        
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        
        while run:
            _, frame = camera.read()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = detect_objects(frame.copy(), net, classes, colors)
                FRAME_WINDOW.image(processed_frame)
            else:
                st.write("Error: Could not access webcam")
                break
            
            time.sleep(0.1)  # Add small delay to reduce CPU usage
        
        camera.release()

if __name__ == '__main__':
    main()