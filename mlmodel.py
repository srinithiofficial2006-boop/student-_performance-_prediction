import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("student_performance_prediction_dataset-2.csv")

# Features
X = df[
    [
        "study_hours",
        "attendance",
        "sleep_hours",
        "previous_grade",
        "assignments_completed",
        "practice_tests_taken"
    ]
]

# Target
y = df["final_grade"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Saved Successfully")