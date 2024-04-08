from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from joblib import load

# Replaceable variables
template_folder = 'templates'

app = Flask(__name__, template_folder=template_folder)

# Load the saved model
model = load('Random_forest_model.joblib')

# Preprocessing function
def preprocess_input(gender, age, occupation, sleep_duration, quality_of_sleep, physical_activity, stress, bmi_category, systolic, diastolic, heart_rate, daily_step):
    # Preprocess categorical features
    gender_encoded = 1 if gender == 'Male' else 0
    occupation_encoded = {'Engineer': 0, 'Doctor': 1, 'Teacher': 2}.get(occupation, 0)
    bmi_category_encoded = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}.get(bmi_category, 1)
    
    # Convert sleep duration to hours
    sleep_duration = round(sleep_duration)  # Round to nearest integer
    # Convert physical activity scale to percentage
    physical_activity_percent = physical_activity / 100.0

    # Return preprocessed input
    return [gender_encoded, age, occupation_encoded, sleep_duration, quality_of_sleep, physical_activity_percent, stress, bmi_category_encoded, systolic, diastolic, heart_rate, daily_step]

# Map numeric labels to string labels
label_mapping = {0: "None", 1: "Sleep Apnea", 2: "Insomnia"}

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input
        gender = request.form['gender']
        age = int(request.form['age'])
        occupation = request.form['occupation']
        sleep_duration = float(request.form['sleep_duration'])
        quality_of_sleep = int(request.form['quality_of_sleep'])
        physical_activity = int(request.form['physical_activity'])
        stress = int(request.form['stress'])
        bmi_category = request.form['bmi_category']
        systolic = int(request.form['systolic'])
        diastolic = int(request.form['diastolic'])
        heart_rate = int(request.form['heart_rate'])
        daily_step = int(request.form['daily_step'])
        
        # Preprocess input
        input_data = preprocess_input(gender, age, occupation, sleep_duration, quality_of_sleep, physical_activity, stress, bmi_category, systolic, diastolic, heart_rate, daily_step)
        # Make prediction
        prediction = model.predict([input_data])[0]
        # Convert numeric label to string label
        predicted_label = label_mapping.get(prediction)
        
        return render_template('result.html', predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
