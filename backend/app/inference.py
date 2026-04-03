# inference.py

# ---------------------------------
# IMPORTS
# ---------------------------------

# io helps us read uploaded image bytes in memory.
import io

# numpy helps with arrays and prediction processing.
import numpy as np

# PIL is used to open and process images.
from PIL import Image

# Import the already-loaded model and metadata.
from app.model_loader import model, metadata


# ---------------------------------
# READ VALUES FROM METADATA
# ---------------------------------

# Read image size from metadata.
# If missing, default to 224.
IMAGE_SIZE = metadata.get("IMAGE_SIZE", 224)

# Read class names from metadata.
# If missing, use fallback class names.
CLASS_NAMES = metadata.get("class_names", ["Bleeding", "Ischemia", "Normal"])


# ---------------------------------
# IMAGE PREPROCESSING FUNCTION
# ---------------------------------

# This function prepares uploaded image bytes for prediction.
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    # Open the uploaded image from raw bytes.
    image = Image.open(io.BytesIO(image_bytes))

    # Convert image to RGB so it always has 3 channels.
    image = image.convert("RGB")

    # Resize image to the input size expected by the model.
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert image to NumPy array as float32.
    # Shape becomes: (224, 224, 3)
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension so shape becomes: (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)

    # IMPORTANT:
    # We do NOT call tf.keras.applications.resnet.preprocess_input here,
    # because the saved inference model already contains the preprocessing
    # layer inside the model itself.
    #
    # So we only convert, resize, and batch the image here.

    return image_array


# ---------------------------------
# PROBABILITY SAFETY FUNCTION
# ---------------------------------

# This function makes sure the output behaves like probabilities.
def ensure_probabilities(prediction_vector: np.ndarray) -> np.ndarray:
    # Flatten output to 1D array.
    prediction_vector = np.asarray(prediction_vector).reshape(-1)

    # If empty, return as is.
    if prediction_vector.size == 0:
        return prediction_vector

    # Check whether values already look like probabilities.
    is_probability_like = (
        prediction_vector.min() >= 0.0
        and prediction_vector.max() <= 1.0
        and np.isclose(prediction_vector.sum(), 1.0, atol=1e-2)
    )

    # If not, apply softmax manually.
    if not is_probability_like:
        exp_values = np.exp(prediction_vector - np.max(prediction_vector))
        prediction_vector = exp_values / np.sum(exp_values)

    return prediction_vector


# ---------------------------------
# MAIN PREDICTION FUNCTION
# ---------------------------------

# This function takes image bytes and returns prediction results.
def predict_image(image_bytes: bytes) -> dict:
    # Preprocess image bytes.
    processed_image = preprocess_image(image_bytes)

    # Run prediction using the loaded model.
    raw_prediction = model.predict(processed_image, verbose=0)

    # Convert output to NumPy array and remove extra dimensions.
    prediction_vector = np.asarray(raw_prediction).squeeze()

    # If output becomes scalar by mistake, convert it into array.
    if prediction_vector.ndim == 0:
        prediction_vector = np.array([float(prediction_vector)])

    # Ensure prediction is in probability format.
    probabilities = ensure_probabilities(prediction_vector)

    # If class count does not match output length,
    # use generic fallback names instead of crashing.
    if len(probabilities) != len(CLASS_NAMES):
        class_names = [f"Class_{i}" for i in range(len(probabilities))]
    else:
        class_names = CLASS_NAMES

    # Find index of highest probability.
    top_index = int(np.argmax(probabilities))

    # Get predicted class name.
    predicted_class = class_names[top_index]

    # Get confidence score.
    confidence = float(probabilities[top_index])

    # Create class-to-probability mapping.
    probabilities_dict = {
        class_names[i]: float(probabilities[i])
        for i in range(len(probabilities))
    }

    # Return final result dictionary.
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities_dict,
        "message": "Prediction completed successfully"
    }