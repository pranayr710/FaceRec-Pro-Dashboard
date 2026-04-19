import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model # Import Model for functional API
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D # GlobalAveragePooling2D is useful for transfer learning
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.applications import VGG16, ResNet50 # We'll use VGG16 or ResNet50 as base
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight
import math

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')
MODELS_DIR = os.path.join(BASE_DIR, '../models')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'emotion_transfer_learning_model.h5') # New model name
HISTORY_PLOT_PATH = os.path.join(MODELS_DIR, 'training_history_transfer_learning.png') # New plot name

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Define image parameters
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64
NUM_CLASSES = 7

# VGG16 was trained on RGB images of size 224x224 or larger.
# Our images are 48x48 grayscale. We need to adapt.
# Option 1: Use a model that can handle smaller input or resize images.
# Option 2: Convert grayscale to RGB (3 channels) by stacking the channel.

# For VGG16, it expects (224, 224, 3) as input typically.
# Resizing our 48x48 images to 224x224 can be done by ImageDataGenerator.
# Converting grayscale to 3 channels can also be done within the generator or model preprocessing.

# Let's adjust target_size for transfer learning. 48x48 is too small for many pre-trained models.
# ResNet50 is more flexible with input sizes, but usually 197x197 or 224x224.
# Let's aim for 96x96 for a balance if we can. If not, 224x224.
# For FER-2013, 48x48 is fixed, so we must resize.

# Data Augmentation and Preprocessing for Transfer Learning
# Pre-trained models expect 3 channels (RGB). We will convert grayscale to 3 channels.
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

print("Loading training data for Transfer Learning...")
# Note: color_mode='rgb' means ImageDataGenerator will convert grayscale to RGB by duplicating the channel
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH), # Keep 48x48, but pre-trained model will require upsampling or special input
    batch_size=BATCH_SIZE,
    color_mode='rgb', # Convert to 3 channels for pre-trained models
    class_mode='categorical',
    subset='training'
)

print("Loading validation data for Transfer Learning...")
validation_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    subset='validation'
)

print("Loading test data for Transfer Learning...")
test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)

# Calculate Class Weights
print("\nCalculating class weights for imbalanced dataset...")
y_train_indices = train_generator.classes
computed_class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_indices),
    y=y_train_indices
)
class_weights_dict = {i: computed_class_weights[i] for i in range(len(computed_class_weights))}
print("Class weights calculated:", class_weights_dict)

# --- Build the Transfer Learning Model ---
print("Building the Transfer Learning model (VGG16 as base)...")

# Load VGG16 pre-trained on ImageNet, excluding the top (classification) layer
# VGG16 expects input shape of at least (48, 48, 3) and typically (224, 224, 3) for its original training.
# However, for smaller custom sizes, we can set include_top=False and input_shape to our target (48, 48, 3).
# Keras automatically handles the necessary adaptions for the convolutional layers if the aspect ratio is maintained.
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the layers of the base model
# This means their weights will not be updated during training.
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x) # Reduces spatial dimensions to a single vector
x = Dense(256, activation='relu')(x) # A new dense layer
x = BatchNormalization()(x)
x = Dropout(0.5)(x) # Strong dropout for new layers
predictions = Dense(NUM_CLASSES, activation='softmax')(x) # Output layer

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model (only new layers are trainable)
initial_learning_rate = 0.001
optimizer = Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# --- First Training Phase: Train only the top (new) layers ---
print("\nStarting First Training Phase: Training only the custom top layers...")
EPOCHS_PHASE1 = 20 # Train for fewer epochs initially
history_phase1 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS_PHASE1,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    class_weight=class_weights_dict,
    callbacks=[
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
    ]
)

print("\nFirst Training Phase complete. Unfreezing some layers for fine-tuning...")

# --- Second Training Phase: Fine-tuning ---
# Unfreeze some layers of the base model for fine-tuning
# It's common to unfreeze the last few convolutional blocks
for layer in base_model.layers[-4:]: # Unfreeze the last 4 layers of VGG16, adjust as needed
    layer.trainable = True

# Re-compile the model with a much lower learning rate for fine-tuning
# This is crucial to avoid destroying the pre-trained weights
fine_tune_learning_rate = initial_learning_rate / 10 # or initial_learning_rate / 100
optimizer_fine_tune = Adam(learning_rate=fine_tune_learning_rate)
model.compile(optimizer=optimizer_fine_tune, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print("\nStarting Second Training Phase: Fine-tuning the entire model (with a very low learning rate)...")
EPOCHS_PHASE2 = 50 # Additional epochs for fine-tuning
total_epochs = EPOCHS_PHASE1 + EPOCHS_PHASE2

# Continue training from the state after phase 1
history_phase2 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS_PHASE2,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    class_weight=class_weights_dict,
    initial_epoch=history_phase1.epoch[-1] + 1, # Start from where phase 1 left off
    callbacks=[
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1), # Adjusted patience
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.000001, verbose=1) # Adjusted min_lr
    ]
)

print("\nModel training complete.")
print(f"Best model saved to: {MODEL_SAVE_PATH}")

# --- Evaluate the model on the test set ---
print("\nEvaluating model on the test set...")
best_model = tf.keras.models.load_model(MODEL_SAVE_PATH) # Load the best model saved
loss, accuracy = best_model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# --- Plot training history for both phases ---
def plot_combined_history(history1, history2, save_path):
    plt.figure(figsize=(14, 6))

    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

plot_combined_history(history_phase1, history_phase2, HISTORY_PLOT_PATH)
print(f"Training history plot saved to: {HISTORY_PLOT_PATH}")