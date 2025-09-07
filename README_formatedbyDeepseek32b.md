# Rotten Fruit Classifier 🍎🤢

A web-based application that uses a deep learning model to classify images of fruits as either fresh or rotten.
You can either upload an image or use your webcam for real-time classification.

## Features ✨
- **Image Classification**: Distinguishes between fresh and rotten fruits.
- **File Upload**: Upload an image file from your local machine for prediction.
- **Webcam Capture**: Capture an image directly from your webcam for instant classification.
- **Confidence Score**: Displays the prediction result along with the model's confidence percentage.
- **Simple UI**: Clean and intuitive user interface built with HTML, CSS, and JavaScript.

## Tech Stack 🛠️
- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras
- **Frontend**: HTML, JavaScript, CSS
- **Core Libraries**: NumPy, Pillow, Scikit-learn

## Project Structure 📂
```
├── static/
│   └── uploads/          # Uploaded images are stored here
├── templates/
│   └── index.html        # Frontend HTML file
├── app.py                # Main Flask application
├── model_training.ipynb  # Jupyter Notebook for model training
├── req.txt               # Python dependencies
└── rotten_fruit_classifier.h5 # The pre-trained model
```

## Setup and Installation 🚀

### Step-by-Step Guide

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/rotten-fruit-classifier.git
   cd rotten-fruit-classifier
   ```

2. **Create a Virtual Environment**
   - For Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - For macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

   > **Why use a virtual environment?**
   > A virtual environment helps manage project-specific dependencies, preventing conflicts between different
projects.

3. **Install Dependencies**
   ```bash
   pip install -r req.txt
   ```
   > **Why these dependencies?**
   > The libraries in `req.txt` are essential for running the application:
     - Flask: Web framework.
     - TensorFlow: For machine learning model loading and predictions.
     - NumPy: For data handling.
     - Pillow: Image processing.

## How to Use 📖

1. **Run the Flask App**
   ```bash
   python app.py
   ```

2. **Open the Application**
   Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. **Make a Prediction**
   - **Option 1: Upload an Image**
     - Click the "Choose File" button.
     - Select an image of a fruit.
     - Click "Upload and Predict".
   - **Option 2: Use Webcam**
     - Allow browser access to your webcam.
     - Point the webcam at a fruit.
     - Click "Capture from Webcam".

4. **View the Result**
   The application displays:
   - Uploaded or captured image.
   - Predicted class (Fresh or Rot).
   - Confidence score.

## Model Details 🧠

### Architecture
The model is a Convolutional Neural Network (CNN) built with TensorFlow and Keras:
1. Three Conv2D layers with ReLU activation, each followed by MaxPooling2D.
2. Flatten layer to convert 2D feature maps to 1D vector.
3. Dense layer with 128 neurons and ReLU activation.
4. Dropout layer with rate of 0.5.
5. Final Dense output layer with 2 neurons and softmax activation.

### Training
- **Dataset**: Trained on 23,619 images (training) and validated on 6,738 images.
- **Data Augmentation**: Includes rotations, shifts, shears, zooms, and horizontal flips.
- **Performance**: Achieved 91.51% test accuracy after 10 epochs.
