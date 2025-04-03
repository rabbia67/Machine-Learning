"""
CIFAR-10 CNN Classifier with Early Stopping

This script demonstrates the development of a Convolutional Neural Network (CNN)
to classify images from the CIFAR-10 dataset. The process involves:
1. Loading and preprocessing the dataset.
2. Building a CNN model with multiple convolutional blocks, including batch normalization,
   max pooling, and dropout for regularization.
3. Compiling and training the model using data augmentation and early stopping to prevent overfitting.
4. Evaluating the model by plotting loss/accuracy curves and generating a confusion matrix.

Each section of the code is documented with detailed explanations.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Step 1: Load and Preprocess the CIFAR-10 Dataset
# ---------------------------------------------------------------------------
"""
Load the CIFAR-10 dataset, normalize image pixel values, and convert labels to one-hot encoding.
- The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes.
- Pixel values are originally in the range [0, 255] and are normalized to [0, 1].
- Labels are converted from integer values to one-hot encoded vectors, which is required
  for categorical crossentropy loss in multi-class classification.
"""

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize images to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert class labels to one-hot encoding
y_train_cat = utils.to_categorical(y_train, 10)
y_test_cat = utils.to_categorical(y_test, 10)

# ---------------------------------------------------------------------------
# Step 2: Build the CNN Model
# ---------------------------------------------------------------------------
"""
Construct a CNN model using TensorFlow's Keras API with the following architecture:
- Three convolutional blocks:
  * Each block consists of two convolutional layers with ReLU activation, batch normalization,
    and a max pooling layer to reduce spatial dimensions.
  * Dropout is applied after pooling to help mitigate overfitting.
- After the convolutional blocks:
  * A flatten layer converts the 2D feature maps into a 1D vector.
  * A fully connected dense layer further processes the features.
  * The final dense layer uses softmax activation to output class probabilities.
"""

model = models.Sequential([
    # First convolutional block: 2 convolutional layers -> batch normalization -> max pooling -> dropout
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:]),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Second convolutional block: increase filter count to 64
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Third convolutional block: increase filter count to 128
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Fully connected layers: flattening, dense layer, and output layer
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Display the model architecture summary
model.summary()

# ---------------------------------------------------------------------------
# Step 3: Compile and Train the Model with Early Stopping
# ---------------------------------------------------------------------------
"""
Compile the model and prepare it for training:
- The model is compiled using the Adam optimizer, which adjusts the learning rate adaptively.
- Categorical crossentropy is used as the loss function since this is a multi-class classification problem.
- Data augmentation via ImageDataGenerator is applied to increase the diversity of the training set.
- Early Stopping is employed to monitor the validation loss and halt training if it doesn't improve
  for a specified number of epochs (patience), thereby preventing overfitting and saving time.
"""

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation to enrich the training dataset
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Define early stopping callback: monitor validation loss with a patience of 5 epochs.
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define training parameters
batch_size = 64
epochs = 50

# Train the model using the augmented data generator and early stopping callback
history = model.fit(datagen.flow(x_train, y_train_cat, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test_cat),
                    callbacks=[early_stop])

# ---------------------------------------------------------------------------
# Step 4: Evaluate Model Performance
# ---------------------------------------------------------------------------
"""
Evaluate the trained model on the test dataset:
- The evaluation includes measuring the test accuracy and loss.
- Loss and accuracy curves are plotted to visualize the training history.
- A confusion matrix is generated to identify the classes that are correctly or incorrectly predicted.
"""

# Evaluate the model on the test set and print the test accuracy
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=2)
print("Test accuracy:", test_acc)

# Plot training and validation loss and accuracy curves
plt.figure(figsize=(12, 5))

# Plot Loss Curves
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy Curves
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Generate predictions for the test set to build the confusion matrix
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test.flatten()

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10))
plt.yticks(tick_marks, range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
