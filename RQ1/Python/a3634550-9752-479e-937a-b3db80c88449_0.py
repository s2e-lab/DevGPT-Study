from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model only once when the app starts
model = tf.keras.models.load_model('digit_recognizer_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the request contains a file
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        # If the user does not select a file, the browser may submit an empty part without a filename
        if file.filename == '':
            return 'No selected file'

        # Save the uploaded file to a specific location (you can customize this)
        file.save('uploads/' + file.filename)

        # Read the image using OpenCV
        img = cv2.imread('uploads/' + file.filename, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (28, 28))
        img_inverted = np.invert(img_resized)
        img_expanded = np.expand_dims(img_inverted, axis=0)
        img_expanded = np.expand_dims(img_expanded, axis=-1)

        # Make a prediction using the loaded model
        prediction = model.predict(img_expanded)
        predicted_digit = int(np.argmax(prediction))

        # Remove the uploaded file from the server (optional)
        os.remove('uploads/' + file.filename)

        return jsonify({'prediction': predicted_digit})

if __name__ == '__main__':
    app.run(debug=True)
