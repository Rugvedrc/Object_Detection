# YOLO Object Detection App

A real-time object detection application built with Streamlit and YOLOv3. This application supports both image upload and webcam-based object detection, making it versatile for various use cases.

## Features

- 🖼️ Image Upload Detection: Upload images and detect objects
- 📹 Webcam Detection: Real-time object detection using your webcam
- ⚙️ Configurable Model Settings: Easy model configuration through the sidebar
- 🎯 High Accuracy: Uses YOLOv3 for accurate object detection
- 🚀 Real-time Processing: Efficient processing with non-maximum suppression
- 📊 Confidence Scores: Displays detection confidence for each object

## Prerequisites

Before running this application, make sure you have the following:

- Python 3.7 or higher
- Required YOLOv3 files:
  - `yolov3.cfg` (Configuration file)
  - `yolov3.weights` (Pre-trained weights)
  - `yolov3.txt` (Class names file)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yolo-object-detection-app.git
cd yolo-object-detection-app
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download YOLOv3 weights:
```bash
wget https://pjreddie.com/media/files/yolov3.weights
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Configure the model paths in the sidebar:
   - Config Path: Path to your `yolov3.cfg` file
   - Weights Path: Path to your `yolov3.weights` file
   - Classes Path: Path to your `yolov3.txt` file

3. Choose your detection mode:
   - Image Upload: Upload an image for object detection
   - Webcam: Use your webcam for real-time detection

## Project Structure

```
yolo-object-detection-app/
│
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── yolov3.cfg            # YOLOv3 configuration file
├── yolov3.weights        # YOLOv3 pre-trained weights
├── yolov3.txt            # Class names file
└── README.md             # Project documentation
```

## Dependencies

- streamlit
- opencv-python
- numpy

## Requirements

Create a `requirements.txt` file with the following contents:

```
streamlit==1.28.0
opencv-python==4.8.1.78
numpy==1.24.3
```

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLO (You Only Look Once) object detection system
- Streamlit for the web interface
- OpenCV for image processing

## Support

For support, please open an issue in the GitHub repository or contact [rugved.rc1@gmail.com].