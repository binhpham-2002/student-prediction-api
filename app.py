from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("student_score_pipeline.pkl")

required_features = [
    'Hours_Studied','Attendance','Sleep_Hours','Previous_Scores',
    'Tutoring_Sessions','Physical_Activity','Parental_Involvement',
    'Access_to_Resources','Extracurricular_Activities','Motivation_Level',
    'Internet_Access','Family_Income','School_Type','Peer_Influence',
    'Learning_Disabilities','Gender'
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        input_data = {feature: request.form.get(feature) for feature in required_features}
        input_df = pd.DataFrame([input_data])
        prediction = round(model.predict(input_df)[0], 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
