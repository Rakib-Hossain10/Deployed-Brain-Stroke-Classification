# inference.py

# ---------------------------------
# IMPORTS
# ---------------------------------

# io helps us handle uploaded image bytes in memory.
import io

# numpy is used for arrays and prediction processing.
import numpy as np

# tensorflow is needed for preprocessing and model prediction.
import tensorflow as tf

# PIL helps us open and process images safely.
from PIL import Image

# We import the already loaded model and metadata from model_loader.py.
# This means the model is loaded once and reused.
from app.model_loader import model, metadata


# ---------------------------------
# READ IMPORTANT VALUES FROM METADATA
# ---------------------------------

# Get the image size from metadata.
# If IMAGE_SIZE does not exist in metadata, use 224 as a safe default.
IMAGE_SIZE = metadata.get("IMAGE_SIZE", 224)

# Get the class names from metadata.
# If class_names does not exist, use your known class order as fallback.
CLASS_NAMES = metadata.get("class_names", ["Bleeding", "Ischemia", "Normal"])


# ---------------------------------
# IMAGE PREPROCESSING FUNCTION
# ---------------------------------

# This function takes raw image bytes from the uploaded file
# and converts them into the exact format expected by your model.
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    # Open the uploaded image from bytes using PIL.
    image = Image.open(io.BytesIO(image_bytes))

    # Convert the image to RGB.
    # This is important because some images may be grayscale or RGBA.
    image = image.convert("RGB")

    # Resize the image to the input size expected by your model.
    # Metadata says IMAGE_SIZE is 224, so this becomes (224, 224).
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert the PIL image into a NumPy array of type float32.
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension.
    # Model expects shape like: (1, 224, 224, 3)
    # Instead of: (224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)

    # Apply ResNet50 preprocessing.
    # This is VERY important because your training code used:
    # tf.keras.applications.resnet.preprocess_input
    # So inference should use the same preprocessing.
    image_array = tf.keras.applications.resnet.preprocess_input(image_array)

    # Return the processed image array.
    return image_array


# ---------------------------------
# SOFTMAX SAFETY FUNCTION
# ---------------------------------

# This function makes sure prediction values behave like probabilities.
# If they are already probabilities, it returns them unchanged.
# If they are raw logits, it converts them using softmax.
def ensure_probabilities(prediction_vector: np.ndarray) -> np.ndarray:
    # Convert input into a flat NumPy array.
    prediction_vector = np.asarray(prediction_vector).reshape(-1)

    # If the vector is empty, return it as is.
    if prediction_vector.size == 0:
        return prediction_vector

    # Check if values already look like probabilities:
    # - minimum should not be below 0
    # - maximum should not be above 1
    # - sum should be close to 1
    is_probability_like = (
        prediction_vector.min() >= 0.0
        and prediction_vector.max() <= 1.0
        and np.isclose(prediction_vector.sum(), 1.0, atol=1e-2)
    )

    # If not probability-like, apply softmax manually.
    if not is_probability_like:
        exp_values = np.exp(prediction_vector - np.max(prediction_vector))
        prediction_vector = exp_values / np.sum(exp_values)

    # Return probability vector.
    return prediction_vector


# ---------------------------------
# MAIN PREDICTION FUNCTION
# ---------------------------------

# This function takes uploaded image bytes and returns prediction results.
def predict_image(image_bytes: bytes) -> dict:
    # Preprocess the uploaded image.
    processed_image = preprocess_image(image_bytes)

    # Run model prediction.
    # verbose=0 means no extra TensorFlow logs in terminal.
    raw_prediction = model.predict(processed_image, verbose=0)

    # Convert prediction to NumPy array and remove unnecessary dimensions.
    prediction_vector = np.asarray(raw_prediction).squeeze()

    # If prediction becomes a single scalar by mistake, wrap it into an array.
    if prediction_vector.ndim == 0:
        prediction_vector = np.array([float(prediction_vector)])

    # Make sure output is in probability form.
    probabilities = ensure_probabilities(prediction_vector)

    # If model output size does not match class name count,
    # create generic class names to avoid crash.
    if len(probabilities) != len(CLASS_NAMES):
        class_names = [f"Class_{i}" for i in range(len(probabilities))]
    else:
        class_names = CLASS_NAMES

    # Find the index of the highest probability.
    top_index = int(np.argmax(probabilities))

    # Use that index to get predicted class name.
    predicted_class = class_names[top_index]

    # Get the highest probability as confidence score.
    confidence = float(probabilities[top_index])

    # Build dictionary of all class probabilities.
    # Example:
    # {
    #   "Bleeding": 0.05,
    #   "Ischemia": 0.90,
    #   "Normal": 0.05
    # }
    probabilities_dict = {
        class_names[i]: float(probabilities[i])
        for i in range(len(probabilities))
    }

    # Return final result in dictionary format.
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities_dict,
        "message": "Prediction completed successfully"
    }