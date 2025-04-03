# 🧠 CIFAR-10 CNN Classifier with Early Stopping
This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 categories (airplanes, cars, birds, etc.). The model incorporates data augmentation and early stopping to enhance performance and prevent overfitting.

# 🚀 Features
- CNN architecture with 3 convolutional blocks
- Batch normalization, max pooling, and dropout regularization
- Data augmentation for improved generalization
- Early stopping to avoid overfitting
- Visualization of training curves and confusion matrix
# 🗂️ Dataset
The model uses the CIFAR-10 dataset, which includes:
- 50,000 training images
- 10,000 test images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# 🏗️ Model Architecture
- Input Layer
- Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPooling → Dropout
- Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPooling → Dropout
- Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPooling → Dropout
- Flatten
- Dense(128) → BatchNorm → Dropout
- Dense(10, activation='softmax')
# 📦 Requirements
Install dependencies using pip:
<pre>  pip install tensorflow matplotlib scikit-learn  </pre>
# ▶️ Running the Code
python cnn_for_cifar_data.py
# 📊 Training Details
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Batch Size: 64
- Epochs: Up to 50 (with EarlyStopping patience of 5)
- Augmentations:
- Rotation up to 15°
- Horizontal flip
- Width and height shift up to 10%
# 📈 Evaluation
After training, the script:
- Prints test accuracy
- Plots loss and accuracy curves
- Displays a confusion matrix

# 📌 Results Example
Accuracy: ~80–85% on test data (may vary depending on training run)
# Plots:
- 📉 Training vs Validation Loss
- 📈 Training vs Validation Accuracy
- 🔢 Confusion Matrix for class-wise prediction performance
#  📬 Contact
For questions or feedback, feel free to reach out via GitHub issues or email. Email: raabi.waheed@gmail.com
