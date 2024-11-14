from flask import Flask, render_template, request
import joblib
import numpy as np
import plotly
import plotly.graph_objects as go

app = Flask(__name__)

# Load pre-trained model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

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

        # Create Plotly chart
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Hours Remaining"}
            )
        )
        
        # Convert Plotly figure to HTML div
        chart_div = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')

        return render_template('result.html', prediction=round(prediction, 2), chart_div=chart_div)

if __name__ == '__main__':
    app.run(debug=True)
