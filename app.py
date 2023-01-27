import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model

SEQUENCE_LENGTH = 40

# Load the pre-trained model
model = load_model('trained_model.h5')

# Initialize webcam using OpenCV
cap = cv2.VideoCapture(0)

# Set the video frame width and height
cap.set(3, 640)
cap.set(4, 480)

# Initialize variables for processing video
last_time = time.time()
current_time = time.time()

labels = ['Ollie', 'Kickflip', 'Shuvit']

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to the input shape of the model
    gray = cv2.resize(gray, (SEQUENCE_LENGTH, 2048))

    # Normalize the pixel values
    gray = gray / 255

    # Add an extra dimension to the frame for the model
    gray = np.expand_dims(gray, axis=0)

    # Get the current time
    current_time = time.time()

    # Check if five seconds have passed
    if current_time - last_time >= 5:
        # Predict the trick using the model
        prediction = model.predict(gray)

        # Get the index of the highest probability
        index = np.argmax(prediction)

        # Get the label of the predicted trick
        label = labels[index]

        # Print the predicted trick
        print("Predicted trick: " + label)

        # Update the last time
        last_time = current_time

# Release the webcam
cap.release()
