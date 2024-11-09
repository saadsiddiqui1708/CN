from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from form
        manpower_hours = float(request.form['manpower_hours'])
        equipment_hours = float(request.form['equipment_hours'])
        project_stage = float(request.form['project_stage'])
        
        # Prepare input for prediction
        features = np.array([[manpower_hours, equipment_hours, project_stage]])
        features_scaled = scaler.transform(features)
        
        # Predict using the loaded model
        prediction = model.predict(features_scaled)[0]

        # Generate plot
        plt.figure(figsize=(6, 4))
        plt.bar(['Manpower Hours', 'Equipment Hours', 'Project Stage'], 
                [manpower_hours, equipment_hours, project_stage*100], color=['blue', 'green', 'red'])
        plt.xlabel('Feature')
        plt.ylabel('Value')
        plt.title('Input Features')
        plt.savefig('static/plot.png')
        plt.close()

        return render_template('result.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
