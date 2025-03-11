from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Define the input structure using Pydantic
class PlayerData(BaseModel):
    appearance: int
    minutes_played: int
    games_injured: int
    award: int
    highest_value: float

# Load the KNN model
modelKnn = joblib.load('Models/knn_model.joblib')  # Corrected model path

# Load the scaler
scaler_classification = joblib.load('Models/scaler.joblib')  # Corrected scaler path

# Define the prediction endpoint
@app.post("/predict/")
def predict(player_data: PlayerData):
    try:
        # Convert the input data into a numpy array for the model
        input_data = np.array([
            player_data.appearance,
            player_data.minutes_played,
            player_data.games_injured,
            player_data.award,
            player_data.highest_value
        ]).reshape(1, -1)  # Reshape to a 2D array for the model

        # Scale the input data
        scaled_input = scaler_classification.transform(input_data)

        # Make a prediction
        prediction = modelKnn.predict(scaled_input)

        # Return the prediction as a standard Python int for compatibility with FastAPI
        return {"prediction": int(prediction[0])}  # Convert to native int

    except Exception as e:
        # Log the error and return a 500 Internal Server Error response
        print(f"Error: {e}")
        return {"error": "Internal Server Error", "details": str(e)}

# Optionally, add a basic endpoint for testing the server
@app.get("/")
def read_root():
    return {"message": "Welcome to the KNN model prediction API"}
