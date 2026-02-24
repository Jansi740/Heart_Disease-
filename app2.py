from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)

model=pickle.load(open("heart_model.pkl","rb"))
@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict",methods=["POST"])
def predict():
    age = float(request.form["age"])
    sex = float(request.form["sex"])
    cp = float(request.form["chest_pain_type"])
    trestbps = float(request.form["resting_bp_s"])
    chol = float(request.form["cholesterol"])
    fbs = float(request.form["fasting_blood_sugar"])
    restecg = float(request.form["resting_ecg"])
    thalach = float(request.form["max_heart_rate"])
    exang = float(request.form["exercise_angina"])
    oldpeak = float(request.form["oldpeak"])
    slope = float(request.form["ST_slope"])
    features=np.array([age, sex, cp, trestbps, chol,
       fbs, restecg, thalach,
       exang, oldpeak, slope]).reshape(1, -1)
    prediction=model.predict(features)

    if prediction[0] ==1:
        result=("Patient likely to have disease ")
    else:
        result=("Patient likely to be not diseased")
    return render_template("form.html",prediction_text=result)

if __name__=="__main__":
    app.run(debug=True)
