from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.pkl not found. Run mlmodel.py first.")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
      
        features = [
            float(request.form["study_hours"]),
            float(request.form["attendance"]),
            float(request.form["sleep_hours"]),
            float(request.form["previous_grade"]),
            float(request.form["assignments_completed"]),
            float(request.form["practice_tests_taken"])
        ]

       
        final_features = np.array([features])

      
        prediction = model.predict(final_features)

        return render_template(
            "index.html",
            prediction=round(float(prediction[0]), 2)
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)