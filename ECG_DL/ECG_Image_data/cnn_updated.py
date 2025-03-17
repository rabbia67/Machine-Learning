import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import time
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

# Define paths for local environment (adjust if needed)
GDRIVE_PATH = "D:\\Downloads\\ECG_DL\\ECG_Image_data"
TRAIN_PATH = os.path.join(GDRIVE_PATH, "train")
TEST_PATH = os.path.join(GDRIVE_PATH, "test")
MODEL_SAVE_PATH = os.path.join(GDRIVE_PATH, "models")
RESULTS_PATH = os.path.join(GDRIVE_PATH, "results")
BINARY_TRAIN_PATH = os.path.join(GDRIVE_PATH, "binary_train")
BINARY_TEST_PATH = os.path.join(GDRIVE_PATH, "binary_test")

# Constants
IMG_HEIGHT, IMG_WIDTH = 64, 64  # Reduced image size to lower memory usage
BATCH_SIZE = 64  # Reduced batch size to lower memory usage
EPOCHS = 50
NORMAL_CLASS = "N"
ABNORMAL_CLASSES = ["F", "M", "Q", "S", "V"]


def create_directories():
    """
    Create necessary directories.
    """
    directories = [
        MODEL_SAVE_PATH,
        RESULTS_PATH,
        BINARY_TRAIN_PATH,
        BINARY_TEST_PATH,
        os.path.join(BINARY_TRAIN_PATH, "Normal"),
        os.path.join(BINARY_TRAIN_PATH, "Abnormal"),
        os.path.join(BINARY_TEST_PATH, "Normal"),
        os.path.join(BINARY_TEST_PATH, "Abnormal")
    ]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Directory created or exists: {directory}")
        except Exception as e:
            logging.warning(f"Could not create directory {directory}: {e}")


def organize_data():
    """
    Organize data into binary classification folders with error handling.
    """

    def copy_files(source_path, target_path, class_type):
        if not os.path.exists(source_path):
            logging.warning(f"Source path does not exist: {source_path}")
            return

        target_folder = os.path.join(target_path, class_type)
        if not os.path.exists(target_folder):
            try:
                os.makedirs(target_folder, exist_ok=True)
                logging.info(f"Created target folder: {target_folder}")
            except Exception as e:
                logging.error(f"Error creating directory {target_folder}: {e}")
                return

        try:
            files = os.listdir(source_path)
            for file in files:
                source_file = os.path.join(source_path, file)
                target_file = os.path.join(target_folder, file)
                if not os.path.exists(target_file):
                    try:
                        shutil.copy2(source_file, target_file)
                    except Exception as e:
                        logging.error(f"Error copying {file}: {e}")
        except Exception as e:
            logging.error(f"Error accessing directory {source_path}: {e}")

    # Organize training data
    for class_name in [NORMAL_CLASS] + ABNORMAL_CLASSES:
        source_path = os.path.join(TRAIN_PATH, class_name)
        class_type = "Normal" if class_name == NORMAL_CLASS else "Abnormal"
        copy_files(source_path, BINARY_TRAIN_PATH, class_type)

    # Organize test data
    for class_name in [NORMAL_CLASS] + ABNORMAL_CLASSES:
        source_path = os.path.join(TEST_PATH, class_name)
        class_type = "Normal" if class_name == NORMAL_CLASS else "Abnormal"
        copy_files(source_path, BINARY_TEST_PATH, class_type)
    logging.info("Data organization complete.")


def verify_data_organization():
    """
    Verify that data has been organized correctly.
    """
    for path in [BINARY_TRAIN_PATH, BINARY_TEST_PATH]:
        for class_type in ["Normal", "Abnormal"]:
            class_path = os.path.join(path, class_type)
            if os.path.exists(class_path):
                file_count = len(os.listdir(class_path))
                logging.info(f"Found {file_count} files in {class_path}")
            else:
                logging.warning(f"Directory not found: {class_path}")


def check_images(directory, sample_size=1000):
    """
    Check a sample of images for corruption or emptiness with progress tracking.
    """
    start_time = time.time()
    for class_name in ["Normal", "Abnormal"]:
        class_path = os.path.join(directory, class_name)
        if os.path.exists(class_path):
            files = os.listdir(class_path)
            sampled_files = random.sample(files, min(sample_size, len(files)))
            total_files = len(sampled_files)
            logging.info(f"Checking {total_files} sampled files in {class_path}...")
            for i, file in enumerate(sampled_files, 1):
                img_path = os.path.join(class_path, file)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                        if img.size[0] == 0 or img.size[1] == 0:
                            logging.warning(f"Empty or corrupted image: {img_path}")
                    if i % 100 == 0:
                        logging.info(f"Processed {i}/{total_files} files in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    logging.error(f"Error reading image {img_path}: {e}")
            logging.info(f"Finished checking {class_path} in {time.time() - start_time:.2f} seconds")
        else:
            logging.warning(f"Directory not found: {class_path}")


def calculate_class_weights():
    """
    Calculate class weights to handle imbalance, with more emphasis on minority class.
    """
    train_counts = {}
    for folder in ['Normal', 'Abnormal']:
        folder_path = os.path.join(BINARY_TRAIN_PATH, folder)
        train_counts[folder] = len(os.listdir(folder_path)) if os.path.exists(folder_path) else 0

    total = sum(train_counts.values())
    class_weights = {
        0: (1 / train_counts['Normal']) * (total / 2.0) if train_counts['Normal'] != 0 else 0,
        1: (1 / train_counts['Abnormal']) * (total / 1.5) if train_counts['Abnormal'] != 0 else 0
    }
    logging.info(f"Calculated class weights: {class_weights}")
    return class_weights


def create_data_generators():
    """
    Create data generators with enhanced augmentation for better generalization.
    """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    try:
        train_generator = train_datagen.flow_from_directory(
            BINARY_TRAIN_PATH,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training'
        )
        logging.info(f"Train generator samples: {train_generator.samples}")

        validation_generator = train_datagen.flow_from_directory(
            BINARY_TRAIN_PATH,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation'
        )
        logging.info(f"Validation generator samples: {validation_generator.samples}")

        test_generator = test_datagen.flow_from_directory(
            BINARY_TEST_PATH,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )
        logging.info(f"Test generator samples: {test_generator.samples}")

        return train_generator, validation_generator, test_generator

    except Exception as e:
        logging.error(f"Error creating data generators: {e}")
        return None, None, None


def create_balanced_dataset(train_generator):
    """
    Create a balanced dataset using tf.data for memory efficiency.
    This version yields individual samples rather than batches.
    """
    def generator():
        # For each batch yielded by the Keras generator...
        for x, y in train_generator:
            # Yield each image and label individually
            for i in range(x.shape[0]):
                yield x[i], y[i]

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([IMG_HEIGHT, IMG_WIDTH, 3], [])  # Each element is a single image and its label
    )

    # Calculate steps per epoch based on the total number of samples
    steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)

    # Batch the dataset and repeat it for multiple epochs
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.repeat()  # Ensure it can iterate through multiple epochs
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    logging.info("Balanced dataset created using tf.data")
    return dataset, steps_per_epoch


def create_model():
    """
    Create a deeper CNN model for binary classification with more layers and regularization.
    """
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.7),
        BatchNormalization(),
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    logging.info("Model created and compiled.")
    return model


def train_model(model, train_generator, validation_generator, class_weights):
    """
    Train the model with callbacks, class weights, and balanced data.
    """
    try:
        balanced_dataset, steps_per_epoch = create_balanced_dataset(train_generator)
        logging.info("Using balanced dataset for training...")

        # Calculate validation steps
        validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, verbose=1, min_lr=1e-7),
            ModelCheckpoint(filepath=os.path.join(MODEL_SAVE_PATH, 'best_model.h5'),
                            monitor='val_loss', save_best_only=True, verbose=1)
        ]

        history = model.fit(
            balanced_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        logging.info("Training completed.")
        return history

    except Exception as e:
        logging.error(f"Error during training: {e}")
        return None


def evaluate_model(model, test_generator):
    """
    Evaluate the model on test data and print performance metrics.
    Additionally, it plots ROC and AUC curves, box plots, and violin plots.
    """
    try:
        loss, acc = model.evaluate(test_generator)
        logging.info(f"Test Loss: {loss}")
        logging.info(f"Test Accuracy: {acc}")

        # Predict on test data for further evaluation
        test_generator.reset()
        y_pred_prob = model.predict(test_generator, verbose=0)
        y_pred = np.where(y_pred_prob > 0.5, 1, 0)
        y_true = test_generator.classes

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # Classification Report
        logging.info("Classification Report:")
        report = classification_report(y_true, y_pred)
        print(report)

        # ROC Curve and AUC Score
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

        # Box Plot of Predicted Probabilities
        df = pd.DataFrame({'True Label': y_true, 'Predicted Probability': y_pred_prob.flatten()})
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df['True Label'], y=df['Predicted Probability'])
        plt.xlabel('Class (0=Normal, 1=Abnormal)')
        plt.ylabel('Predicted Probability')
        plt.title('Box Plot of Predicted Probabilities')
        plt.xticks(ticks=[0, 1], labels=['Normal', 'Abnormal'])
        plt.show()

        # Violin Plot of Predicted Probabilities
        plt.figure(figsize=(8, 6))
        sns.violinplot(x=df['True Label'], y=df['Predicted Probability'])
        plt.xlabel('Class (0=Normal, 1=Abnormal)')
        plt.ylabel('Predicted Probability')
        plt.title('Violin Plot of Predicted Probabilities')
        plt.xticks(ticks=[0, 1], labels=['Normal', 'Abnormal'])
        plt.show()

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")


def plot_training_history(history):
    """
    Plot training and validation accuracy and loss.
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='upper left')

        # Loss plot
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper left')

        plt.show()
        logging.info("Training history plotted.")
    except Exception as e:
        logging.error(f"Error plotting training history: {e}")


def plot_class_distribution():
    """
    Plot the distribution of classes in the training and testing sets.
    """
    try:
        # Count files in each class folder for training data
        train_counts = {}
        for folder in ['Normal', 'Abnormal']:
            folder_path = os.path.join(BINARY_TRAIN_PATH, folder)
            train_counts[folder] = len(os.listdir(folder_path)) if os.path.exists(folder_path) else 0

        # Count files in each class folder for testing data
        test_counts = {}
        for folder in ['Normal', 'Abnormal']:
            folder_path = os.path.join(BINARY_TEST_PATH, folder)
            test_counts[folder] = len(os.listdir(folder_path)) if os.path.exists(folder_path) else 0

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(x=list(train_counts.keys()), y=list(train_counts.values()), ax=ax[0])
        ax[0].set_title("Train Data Class Distribution")
        ax[0].set_xlabel("Class")
        ax[0].set_ylabel("Number of Images")

        sns.barplot(x=list(test_counts.keys()), y=list(test_counts.values()), ax=ax[1])
        ax[1].set_title("Test Data Class Distribution")
        ax[1].set_xlabel("Class")
        ax[1].set_ylabel("Number of Images")

        plt.show()
        logging.info("Class distribution plotted.")
    except Exception as e:
        logging.error(f"Error plotting class distribution: {e}")


def display_sample_images(num_images=5):
    """
    Display sample images from each class.
    """
    try:
        fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
        for i, class_name in enumerate(["Normal", "Abnormal"]):
            folder = os.path.join(BINARY_TRAIN_PATH, class_name)
            images = os.listdir(folder)[:num_images]
            for j, img in enumerate(images):
                img_path = os.path.join(folder, img)
                image = plt.imread(img_path)
                axes[i, j].imshow(image)
                axes[i, j].axis('off')
                axes[i, j].set_title(class_name)
        plt.show()
        logging.info("Sample images displayed.")
    except Exception as e:
        logging.error(f"Error displaying sample images: {e}")


def main():
    """
    Main execution function with error handling.
    """
    try:
        logging.info("Creating directories...")
        create_directories()

        logging.info("Organizing data...")
        organize_data()

        logging.info("Verifying data organization...")
        verify_data_organization()

        logging.info("Checking image files in training data...")
        check_images(BINARY_TRAIN_PATH, sample_size=1000)
        logging.info("Checking image files in testing data...")
        check_images(BINARY_TEST_PATH, sample_size=1000)

        logging.info("Calculating class weights...")
        class_weights = calculate_class_weights()

        logging.info("Creating data generators...")
        train_generator, validation_generator, test_generator = create_data_generators()
        if None in (train_generator, validation_generator, test_generator):
            logging.error("Failed to create data generators. Exiting...")
            return

        logging.info("Plotting class distributions...")
        plot_class_distribution()
        logging.info("Displaying sample images...")
        display_sample_images(num_images=5)

        logging.info("Creating and training model...")
        model = create_model()
        history = train_model(model, train_generator, validation_generator, class_weights)

        if history is not None:
            logging.info("Plotting training history...")
            plot_training_history(history)

            logging.info("Evaluating model...")
            evaluate_model(model, test_generator)

            logging.info(f"Results (plots and metrics) are now displayed. Model saved in: {MODEL_SAVE_PATH}")
        else:
            logging.error("Training failed. Check logs for errors.")

    except Exception as e:
        logging.error(f"An error occurred in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
