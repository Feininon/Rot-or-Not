from tensorflow.keras.models import load_model
# Load the trained model
model = load_model("rotten_fruit_classifier.h5")
print("Model loaded successfully!")

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize image to match training size
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values (same as training)
    return img_array

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    
    class_labels = ["Fresh", "Rot"]  # Assuming your classes are "fresh" and "rot"
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")
    

image_path = r"orange.jpg"  # Replace with the path to your image
predict_image(image_path)