import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load the large dataset
data = pd.read_csv('large_dataset.csv')

# Define features (X) and target (y)
X = data[['manpower_hours', 'equipment_hours', 'project_stage']]
y = data['hours_remaining']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = RandomForestRegressor()
model.fit(X_scaled, y)

# Save the model and scaler
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')