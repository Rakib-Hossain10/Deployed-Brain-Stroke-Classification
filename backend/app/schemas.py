# schemas.py

# We import BaseModel from pydantic.
# Pydantic is used by FastAPI to define and validate the structure of request/response data.
from pydantic import BaseModel

# We import Dict so we can say that probabilities will be a dictionary like:
# {"Bleeding": 0.12, "Ischemia": 0.80, "Normal": 0.08}
from typing import Dict


# This class defines the JSON response format for your prediction API.
# Whenever your backend predicts an image, it will return data in this structure.
class PredictionResponse(BaseModel):
    # The final predicted class name, for example: "Ischemia"
    predicted_class: str

    # The confidence score of the top predicted class, for example: 0.91
    confidence: float

    # A dictionary containing probabilities for all classes.
    # Example:
    # {
    #   "Bleeding": 0.03,
    #   "Ischemia": 0.91,
    #   "Normal": 0.06
    # }
    probabilities: Dict[str, float]

    # Optional message to show extra info to the frontend or user.
    # Example: "Prediction completed successfully"
    message: str