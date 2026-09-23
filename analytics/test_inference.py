"""
Standalone test script to verify that best_titanic_pipeline.joblib can be loaded
from disk and executed directly on raw, unprocessed input DataFrames.
"""

import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_titanic_pipeline.joblib")


def test_inference_on_raw_data():
    print(f"Loading pipeline from: {MODEL_PATH}")
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"

    pipeline = joblib.load(MODEL_PATH)

    # Test raw DataFrame with missing values and raw types
    raw_data = pd.DataFrame([
        {
            "pclass": 1,
            "sex": "female",
            "age": None,  # Test imputer handling
            "sibsp": 1,
            "parch": 0,
            "fare": 83.15,
            "embarked": "C"
        },
        {
            "pclass": 3,
            "sex": "male",
            "age": 24.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 7.8958,
            "embarked": "S"
        }
    ])

    preds = pipeline.predict(raw_data)
    probs = pipeline.predict_proba(raw_data)[:, 1]

    print("\nInference Results on Raw Input:")
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        label = "Survived" if pred == 1 else "Died"
        print(f"Passenger {i + 1}: Prediction = {label} (Probability = {prob:.4f})")

    assert len(preds) == 2
    print("\nInference test PASSED successfully!")


if __name__ == "__main__":
    test_inference_on_raw_data()
