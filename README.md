# Rotten Fruit Classifier 🍎🤢
--written by AI 

A web-based application that uses a deep learning model to classify images of fruits as either **fresh** or **rotten**. You can either upload an image or use your webcam to capture a picture for real-time classification.


---

## Features ✨

- **Image Classification:** Distinguishes between fresh and rotten fruits.
- **File Upload:** Upload an image file from your local machine for prediction.
- **Webcam Capture:** Capture an image directly from your webcam for instant classification.
- **Confidence Score:** Displays the prediction result along with the model's confidence percentage.
- **Simple UI:** Clean and intuitive user interface built with HTML and JavaScript.

---

## Tech Stack 🛠️

- **Backend:** Python, Flask
- **Machine Learning:** TensorFlow, Keras
- **Frontend:** HTML, JavaScript, CSS
- **Core Libraries:** NumPy, Pillow, Scikit-learn

---

## Project Structure 📂

```

.
├── static/
│   └── uploads/          \# Uploaded images are stored here
├── templates/
│   └── index.html        \# Frontend HTML file
├── app.py                \# Main Flask application
├── model_training.ipynb  \# Jupyter Notebook for model training
├── req.txt               \# Python dependencies
└── rotten_fruit_classifier.h5 \# The pre-trained model

````

---

## Setup and Installation 🚀

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/rotten-fruit-classifier.git](https://github.com/your-username/rotten-fruit-classifier.git)
cd rotten-fruit-classifier
````

### 2\. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

Install all the required Python libraries from the `req.txt` file.

```bash
pip install -r req.txt
```

-----

## How to Use 📖

1.  **Run the Flask App:**
    Execute the `app.py` script from your terminal.

    ```bash
    python app.py
    ```

2.  **Open the Application:**
    Open your web browser and navigate to:
    [http://127.0.0.1:5000](https://www.google.com/search?q=http://127.0.0.1:5000)

3.  **Make a Prediction:**

      * **Option 1: Upload an Image**
          - Click the "Choose File" button.
          - Select an image of a fruit.
          - Click "Upload and Predict".
      * **Option 2: Use Webcam**
          - Allow the browser to access your webcam.
          - Point the webcam at a fruit.
          - Click "Capture from Webcam".

4.  **View the Result:**
    The application will display the uploaded image, the predicted class (Fresh or Rot), and the confidence score.

-----

## Model Details 🧠

The classification model is a Convolutional Neural Network (CNN) built and trained using TensorFlow and Keras.

### Model Architecture

The model consists of:

  - Three **Convolutional layers** (`Conv2D`) with ReLU activation, each followed by a **Max Pooling layer** (`MaxPooling2D`).
  - A **Flatten** layer to convert the 2D feature maps into a 1D vector.
  - A **Dense** layer with 128 neurons and ReLU activation.
  - A **Dropout** layer with a rate of 0.5 to prevent overfitting.
  - A final **Dense** output layer with 2 neurons and a **softmax** activation function for binary classification (Fresh/Rot).

### Training

  - **Dataset:** The model was trained on a dataset of 23,619 images for training and 6,738 images for validation.
  - **Data Augmentation:** To improve robustness, the training data was augmented with random rotations, shifts, shears, zooms, and horizontal flips.
  - **Performance:** The model was trained for 10 epochs and achieved a final **test accuracy of 91.51%**.

--yeah bunch of random stuff is there which are absolutely for fun.

