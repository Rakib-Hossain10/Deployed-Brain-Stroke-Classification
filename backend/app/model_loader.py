# model_loader.py

# -------------------------------
# IMPORTS
# -------------------------------

# os helps us work with file paths like folders and filenames.
import os

# json helps us read the metadata JSON file.
import json

# tensorflow is needed to load your trained .keras model.
import tensorflow as tf


# -------------------------------
# BASE PATH SETUP
# -------------------------------

# __file__ means: the current Python file (model_loader.py).
# os.path.abspath(__file__) gives the full absolute path of this file.
# Example:
# W:\DL FullStack\Brain Stroke\backend\app\model_loader.py
CURRENT_FILE_PATH = os.path.abspath(__file__)

# os.path.dirname(...) gives the folder containing the file.
# So this becomes:
# W:\DL FullStack\Brain Stroke\backend\app
APP_DIR = os.path.dirname(CURRENT_FILE_PATH)

# One level up from app/ is backend/
# This helps us build safe paths without hardcoding full Windows paths.
BACKEND_DIR = os.path.dirname(APP_DIR)

# Inside backend/, you already created a folder called models/
MODELS_DIR = os.path.join(BACKEND_DIR, "models")


# -------------------------------
# FILE PATHS
# -------------------------------

# Full path to the trained Keras model file.
MODEL_PATH = os.path.join(MODELS_DIR, "BrainStroke_resnet50_inference.keras")

# Full path to the metadata JSON file.
METADATA_PATH = os.path.join(MODELS_DIR, "BrainStroke_resnet50_metadata.json")


# -------------------------------
# LOAD MODEL FUNCTION
# -------------------------------

# This function loads the trained model from disk.
# We keep it inside a function so other files can call it when needed.
def load_model():
    # First, check if the model file actually exists.
    # If not, we raise an error with a clear message.
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}"
        )

    # tf.keras.models.load_model(...) loads your saved .keras file.
    model = tf.keras.models.load_model(MODEL_PATH)

    # Return the loaded model so other parts of the app can use it.
    return model


# -------------------------------
# LOAD METADATA FUNCTION
# -------------------------------

# This function reads the JSON metadata file.
# Metadata contains useful info like class names and image size.
def load_metadata():
    # Check if the metadata file exists.
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found at: {METADATA_PATH}"
        )

    # Open the JSON file in read mode with UTF-8 encoding.
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    # Return the metadata dictionary.
    return metadata


# -------------------------------
# LOAD BOTH AT STARTUP
# -------------------------------

# Here we load the model once when this file is imported.
# That means FastAPI does not reload the model again and again for every request.
model = load_model()

# Here we load the metadata once as well.
metadata = load_metadata()