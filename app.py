import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Directory for saving uploads
app.secret_key = 'supersecretkey'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained model
model = tf.keras.models.load_model('saved_model.h5')

def predict_image(image_path, img_size=128):
    # Preprocess the image
    new_img = cv2.imread(image_path)
    if new_img is None:
        raise ValueError(f"Error loading image from {image_path}")
    new_img = cv2.resize(new_img, (img_size, img_size))
    new_img = new_img / 255.0
    new_img = np.expand_dims(new_img, axis=0)

    # Make a prediction
    pred_class = model.predict(new_img)
    predicted_class = np.argmax(pred_class)

    # Return class label based on prediction
    return ["Benign", "Malignant", "Normal"][predicted_class]

# Route to serve files from the 'uploads/' directory
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Main page
@app.route('/')
def index():
    return render_template('index.html')

# Upload and prediction route
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        # Save uploaded file to the uploads folder
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # Predict the class of the uploaded image
        predicted_class = predict_image(image_path)

        # Render the result page
        return render_template('result.html', predicted_class=predicted_class, image_filename=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
