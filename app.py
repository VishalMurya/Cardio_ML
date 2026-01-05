from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model safely
model_path = os.path.join(os.path.dirname(__file__), 'xgb_classifier.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Get form values
    age_years = int(request.form['age_years'])
    gender = int(request.form['gender'])
    height = int(request.form['height'])
    weight = float(request.form['weight'])
    ap_hi = int(request.form['ap_hi'])
    ap_lo = int(request.form['ap_lo'])
    cholesterol = int(request.form['cholesterol'])
    gluc = int(request.form['gluc'])
    smoke = int(request.form['smoke'])
    alco = int(request.form['alco'])
    active = int(request.form['active'])

    # IMPORTANT: convert age to days (as used in training)
    age_days = age_years * 365

    # Arrange features EXACTLY as model was trained
    features = np.array([[ 
        age_days,
        gender,
        height,
        weight,
        ap_hi,
        ap_lo,
        cholesterol,
        gluc,
        smoke,
        alco,
        active
    ]])

    prediction = model.predict(features)[0]

    result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
