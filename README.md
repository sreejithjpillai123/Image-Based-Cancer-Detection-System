Breast Cancer Detection from Ultrasound Images
This project is a breast cancer detection model that leverages machine learning to classify ultrasound images as benign, malignant, or normal. By training on a labeled dataset of ultrasound images, the model can assist in early and accurate breast cancer diagnosis.

Project Overview
The goal of this project is to develop a machine learning model that can classify breast ultrasound images to assist healthcare professionals. It incorporates both classification and segmentation techniques, offering a more interpretable and localized cancer detection approach.

Features
Image Classification: Detects whether an image is benign, malignant, or normal.
Class Activation Maps (CAMs): Highlights image regions that influenced the model’s decision, enhancing interpretability.
Segmentation: Provides a pixel-level estimation of the cancer-affected area for further diagnosis.
Dataset
The dataset used for training and evaluation contains breast ultrasound images and corresponding masks:

Categories: Benign, Malignant, and Normal.
Structure: Each image has an associated mask image indicating regions of interest.
Source: Dataset_BUSI_with_GT (Specify dataset source if publicly available)
Methodology
Data Preprocessing: Image normalization, resizing, and augmentation.
Model Architecture:
CNN for Classification: Built a convolutional neural network to classify images.
UNet for Segmentation: Utilized a U-Net model to identify cancer-affected areas.
Evaluation Metrics: Model accuracy, precision, recall, and segmentation performance metrics.
Results
Classification Accuracy: Achieved an accuracy of X% on the test set.
Segmentation Performance: High overlap with the ground truth masks (IoU Score: X).
Technologies Used
Python: Language for model development.
TensorFlow / Keras: Used for neural network modeling and training.
OpenCV: For image preprocessing.
Flask: Backend for the web interface.

for dataset            https://drive.google.com/file/d/1vTJzXoeIJ8z6n2boLVtyHWk7JFr14u4Y/view?usp=sharing
