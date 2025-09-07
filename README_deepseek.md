# Rotten Fruit Classifier
--written by deepseekv3.1

A web-based application that classifies fruits as fresh or rotten using a Convolutional Neural Network (CNN) model built with TensorFlow/Keras.

## Features

- Upload images for classification
- Real-time webcam capture for instant predictions
- Clean and intuitive web interface
- High accuracy model (91.51% test accuracy)
- Responsive design that works on desktop and mobile devices

## Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Flask (Python web framework)
- **Machine Learning**: TensorFlow, Keras
- **Image Processing**: Pillow, OpenCV (through Keras preprocessing)

## Project Structure

```
rotten-fruit-classifier/
├── app.py                 # Flask application
├── index.html            # Web interface
├── req.txt              # Python dependencies
├── model_training.ipynb  # Jupyter notebook for model training
├── rotten_fruit_classifier.h5  # Trained model (to be generated)
└── static/
    └── uploads/          # Directory for uploaded images
```

## Installation

1. Clone or download the project files
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r req.txt
   ```

## Usage

1. Train the model (optional - if you don't have the .h5 file):
   - Run the Jupyter notebook `model_training.ipynb` to train the model
   - Ensure you have a dataset organized in Train/Test folders with Fresh/Rot subfolders

2. Start the Flask application:
   ```bash
   python app.py
   ```

3. Open your web browser and navigate to `http://localhost:5000`

4. Use the interface to:
   - Upload an image file, or
   - Capture an image using your webcam

5. View the prediction results with confidence percentage

## Model Details

- Architecture: CNN with 3 convolutional layers and 2 dense layers
- Input size: 224×224 pixels
- Classes: Fresh, Rot
- Training accuracy: 92.21%
- Test accuracy: 91.51%
- Training epochs: 10

## Dataset

The model was trained on a dataset containing:
- 23,619 training images
- 6,738 test images
- Balanced between fresh and rotten fruit categories

## Customization

You can modify the model architecture and training parameters in the Jupyter notebook. To retrain with your own dataset, organize your images in the following structure:

```
dataset/
├── Train/
│   ├── Fresh/
│   └── Rot/
└── Test/
    ├── Fresh/
    └── Rot/
```

## License

This project is available for educational and research purposes.
