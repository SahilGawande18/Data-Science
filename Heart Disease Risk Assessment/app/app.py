from flask import Flask, render_template, request, url_for
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('models/heart_disease_prediction_model.pkl', 'rb'))

# Homepage
@app.route('/')
def home():
    return render_template('index.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        chest_pain_type = int(request.form['chest_pain_type'])
        resting_bp = int(request.form['resting_bp'])
        cholesterol = int(request.form['cholesterol'])
        fasting_bs = int(request.form['fasting_bs'])
        resting_ecg = int(request.form['resting_ecg'])
        max_hr = int(request.form['max_hr'])
        exercise_angina = int(request.form['exercise_angina'])
        oldpeak = float(request.form['oldpeak'])
        st_slope = int(request.form['st_slope'])

        # Make prediction
        input_data = [[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]]
        prediction = model.predict(input_data)

        # Process prediction result
        if prediction[0] == 0:
            result = 'Low Risk of Heart Disease'
        else:
            result = 'High Risk of Heart Disease'

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
