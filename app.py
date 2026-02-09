from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model & pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# Ocean proximity → representative geo points
OCEAN_MAP = {
    "NEAR BAY":    {"lat": 37.77, "lon": -122.42},
    "NEAR OCEAN":  {"lat": 34.02, "lon": -118.50},
    "<1H OCEAN":   {"lat": 34.05, "lon": -118.25},
    "INLAND":      {"lat": 36.77, "lon": -119.41},
    "ISLAND":      {"lat": 33.39, "lon": -118.41}
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        ocean = request.form["ocean_proximity"]
        geo = OCEAN_MAP[ocean]

        data = {
            "longitude": geo["lon"],
            "latitude": geo["lat"],
            "housing_median_age": float(request.form["housing_median_age"]),
            "total_rooms": float(request.form["total_rooms"]),
            "total_bedrooms": float(request.form["total_bedrooms"]),
            "population": float(request.form["population"]),
            "households": float(request.form["households"]),
            "median_income": float(request.form["median_income"]),
            "ocean_proximity": ocean
        }

        df = pd.DataFrame([data])
        transformed = pipeline.transform(df)
        prediction = model.predict(transformed)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
