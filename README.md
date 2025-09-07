Rotten Fruit Classifier 🍎🤢A web-based application that uses a deep learning model to classify images of fruits as either fresh or rotten. You can either upload an image or use your webcam to capture a picture for real-time classification.Features ✨Image Classification: Distinguishes between fresh and rotten fruits.File Upload: Upload an image file from your local machine for prediction.Webcam Capture: Capture an image directly from your webcam for instant classification.Confidence Score: Displays the prediction result along with the model's confidence percentage.Simple UI: Clean and intuitive user interface built with HTML, CSS, and JavaScript.Tech Stack 🛠️Backend: Python, FlaskMachine Learning: TensorFlow, KerasFrontend: HTML, JavaScript, CSSCore Libraries: NumPy, Pillow, Scikit-learnProject Structure 📂.
├── static/
│   └── uploads/          # Uploaded images are stored here
├── templates/
│   └── index.html        # Frontend HTML file
├── app.py                # Main Flask application
├── model_training.ipynb  # Jupyter Notebook for model training
├── req.txt               # Python dependencies
└── rotten_fruit_classifier.h5 # The pre-trained model
Setup and Installation 🚀Follow these steps to get the project running on your local machine.1. Clone the Repositorygit clone [https://github.com/your-username/rotten-fruit-classifier.git](https://github.com/your-username/rotten-fruit-classifier.git)
cd rotten-fruit-classifier
2. Create a Virtual EnvironmentIt's recommended to use a virtual environment to manage dependencies.# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
Explanation

🤔 **Why use a virtual environment?**

Think of it as a private, isolated workspace for this specific project. It prevents "dependency conflicts." For example, this project needs `tensorflow==2.10.0`, but another project might need `tensorflow==2.15.0`. A virtual environment creates a self-contained "bubble" so that the libraries for one project don't interfere with another. It keeps your projects organized, self-contained, and reproducible.
3. Install DependenciesInstall all the required Python libraries from the req.txt file.pip install -r req.txt
Explanation

📦 **Why install these specific dependencies?**

The libraries in `req.txt` are the essential building blocks the application needs to run. The code won't work without them.

- **`Flask`**: The web framework used to build the server, handle image uploads, and send back predictions.
- **`tensorflow`**: The core machine learning library used to load your saved model (`rotten_fruit_classifier.h5`) and make predictions.
- **`numpy`**: A library for handling data in arrays. The uploaded image is converted into a NumPy array before being fed into the model.
- **`Pillow`**: An image processing library used to open and resize the images to the format the model expects.

The `req.txt` file is like a shopping list. This command automatically installs all of them for you.
How to Use 📖Run the Flask App:Execute the app.py script from your terminal.python app.py
Open the Application:Open your web browser and navigate to:http://127.0.0.1:5000Make a Prediction:Option 1: Upload an ImageClick the "Choose File" button.Select an image of a fruit.Click "Upload and Predict".Option 2: Use WebcamAllow the browser to access your webcam.Point the webcam at a fruit.Click "Capture from Webcam".View the Result:The application will display the uploaded image, the predicted class (Fresh or Rot), and the confidence score.Model Details 🧠The classification model is a Convolutional Neural Network (CNN) built and trained using TensorFlow and Keras.Model ArchitectureThe model consists of:Three Convolutional layers (Conv2D) with ReLU activation, each followed by a Max Pooling layer (MaxPooling2D).A Flatten layer to convert the 2D feature maps into a 1D vector.A Dense layer with 128 neurons and ReLU activation.A Dropout layer with a rate of 0.5 to prevent overfitting.A final Dense output layer with 2 neurons and a softmax activation function for binary classification (Fresh/Rot).TrainingDataset: The model was trained on a dataset of 23,619 images for training and 6,738 images for validation.Data Augmentation: To improve robustness, the training data was augmented with random rotations, shifts, shears, zooms, and horizontal flips.Performance: The model was trained for 10 epochs and achieved a final test accuracy of 91.51%.
