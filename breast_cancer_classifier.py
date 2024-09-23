import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split

# Use TensorFlow Data API for efficient image loading
def load_images_tf(base_path, img_size, batch_size=16):
    image_paths = []
    labels = []
    categories = ['benign', 'malignant', 'normal']
    label_map = {category: idx for idx, category in enumerate(categories)}

    # Collect image paths and labels
    for category in categories:
        img_dir = os.path.join(base_path, category)
        for img_name in os.listdir(img_dir):
            image_paths.append(os.path.join(img_dir, img_name))
            labels.append(label_map[category])

    # Create a TensorFlow dataset from the image paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def process_image(file_path, label):
        # Load and preprocess the image
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=3)  # Adjust channels to RGB
        img = tf.image.resize(img, [img_size, img_size])
        img = img / 255.0  # Normalize the image
        return img, label

    # Apply image processing function and batch data
    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(image_paths))  # Buffer shuffle for randomness
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset, len(image_paths)

# CNN Model for Classification
def simple_cnn(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)
    
    # Simple CNN layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Classification Output
    classification_output = layers.Dense(3, activation='softmax')(x)
    
    model = Model(inputs, classification_output)
    
    return model

# Train the Model with TensorFlow Dataset API
def train_model(train_dataset, test_dataset, img_size, model_save_path='model.h5'):
    model = simple_cnn(input_size=(img_size, img_size, 3))
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model using datasets
    history = model.fit(train_dataset, 
                        validation_data=test_dataset,
                        epochs=10)  # Reduce epochs for quicker testing
    
    # Save the trained model to a file
    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")
    
    return model

# Load the Saved Model
def load_trained_model(model_save_path='model.h5'):
    # Load the saved model
    if os.path.exists(model_save_path):
        model = tf.keras.models.load_model(model_save_path)
        print(f"Model loaded from {model_save_path}")
    else:
        print(f"No model found at {model_save_path}. Train the model first.")
        model = None
    return model

# Evaluate and Predict
def evaluate_and_predict(model, new_image_path, img_size):
    # Predict on new image
    new_img = cv2.imread(new_image_path)
    new_img = cv2.resize(new_img, (img_size, img_size))
    new_img = new_img / 255.0
    new_img = np.expand_dims(new_img, axis=0)

    pred_class = model.predict(new_img)
    predicted_class = np.argmax(pred_class)

    # Display results
    print(f'Predicted Class: {["Benign", "Malignant", "Normal"][predicted_class]}')

# Main Function
if __name__ == '__main__':
    base_path = 'C:/Users/sreejith/OneDrive/Desktop/cancer/archive/Dataset_BUSI_with_GT'  # Your dataset path
    img_size = 128  # Image size (e.g., 128x128)
    batch_size = 16  # Batch size for TensorFlow dataset
    model_save_path = 'saved_model.h5'  # Path to save/load the model

    # Step 1: Load Data using TensorFlow Data API
    dataset, dataset_size = load_images_tf(base_path, img_size, batch_size=batch_size)

    # Step 2: Split into training and testing datasets
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    # Step 3: Check if a trained model exists, else train the model
    model = load_trained_model(model_save_path)
    if model is None:
        model = train_model(train_dataset, test_dataset, img_size, model_save_path)

    # Step 4: Evaluate and Predict on a new image
    new_image_path = 'C:/Users/sreejith/OneDrive/Desktop/cancer/can.png'  # Specify a new image for prediction
    evaluate_and_predict(model, new_image_path, img_size)
