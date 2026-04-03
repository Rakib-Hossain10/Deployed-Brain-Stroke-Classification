# main.py

# ---------------------------------
# IMPORTS
# ---------------------------------

# FastAPI is the main framework we use to build the backend API.
from fastapi import FastAPI, File, UploadFile, HTTPException

# CORSMiddleware is used to allow your future React frontend
# to talk to this backend from another port/domain.
from fastapi.middleware.cors import CORSMiddleware

# We import the response schema we created in schemas.py.
# This helps FastAPI return a clean and validated JSON response.
from app.schemas import PredictionResponse

# We import the prediction function from inference.py.
# This function will process the uploaded image and return prediction results.
from app.inference import predict_image


# ---------------------------------
# CREATE FASTAPI APP
# ---------------------------------

# This creates your FastAPI application object.
# title, description, and version are shown in Swagger docs (/docs).
app = FastAPI(
    title="Brain Stroke Classification API",
    description="FastAPI backend for brain stroke image classification using ResNet50.",
    version="1.0.0"
)


# ---------------------------------
# ENABLE CORS
# ---------------------------------

# CORS = Cross-Origin Resource Sharing
# Your future React frontend will probably run on:
# http://localhost:3000
# or
# http://127.0.0.1:3000
#
# So we allow those origins here.
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Add CORS middleware to the FastAPI app.
app.add_middleware(
    CORSMiddleware,

    # Allow requests only from the origins listed above.
    allow_origins=origins,

    # Allow cookies/auth if needed later.
    allow_credentials=True,

    # Allow all HTTP methods like GET, POST, OPTIONS, etc.
    allow_methods=["*"],

    # Allow all headers.
    allow_headers=["*"],
)


# ---------------------------------
# ROOT ROUTE
# ---------------------------------

# This is a simple test route.
# When you open http://127.0.0.1:8000/
# it will show a small welcome message.
@app.get("/")
def home():
    return {
        "message": "Brain Stroke Classification API is running successfully."
    }


# ---------------------------------
# HEALTH CHECK ROUTE
# ---------------------------------

# This is another useful route for checking whether the backend is alive.
# Later Render can also use such routes for health checking.
@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


# ---------------------------------
# PREDICTION ROUTE
# ---------------------------------

# This route accepts an uploaded image file and returns prediction results.
# response_model=PredictionResponse means the output must match your schema.
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # First check whether the uploaded file exists.
    if not file:
        raise HTTPException(
            status_code=400,
            detail="No file was uploaded."
        )

    # Check the uploaded file content type.
    # We only allow common image types for now.
    allowed_types = ["image/jpeg", "image/jpg", "image/png"]

    # If uploaded file is not one of the allowed types, raise an error.
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPG, JPEG, or PNG image."
        )

    try:
        # Read the uploaded image as raw bytes.
        image_bytes = await file.read()

        # Send image bytes to the prediction function in inference.py.
        result = predict_image(image_bytes)

        # Return the result.
        # FastAPI will validate it using PredictionResponse schema.
        return result

    except Exception as e:
        # If any unexpected error happens, return HTTP 500.
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )



