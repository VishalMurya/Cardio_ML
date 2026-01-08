import pickle
import numpy as np
from flask import Flask, render_template, request

with open('xgb_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age_years']) * 365

    gender = int(request.form['gender'])   

    height = float(request.form['height'])
    weight = float(request.form['weight'])

    ap_hi = int(request.form['ap_hi'])
    ap_lo = int(request.form['ap_lo'])

    cholesterol = int(request.form['cholesterol'])
    glucose = int(request.form['gluc'])

    smoke = int(request.form['smoke'])
    alcohol = int(request.form['alco'])
    active = int(request.form['active'])

    bmi = weight / ((height / 100) ** 2)

    features = np.array([[ 
        age, gender, height, weight,
        ap_hi, ap_lo,
        cholesterol, glucose,
        smoke, alcohol, active,
        bmi
    ]])

    prediction = model.predict(features)[0]

    result = (
        "High Risk of Heart Disease"
        if prediction == 1
        else " Low Risk of Heart Disease"
    )

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
