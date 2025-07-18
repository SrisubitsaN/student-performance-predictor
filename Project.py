# train_model.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your data
df = pd.read_csv("D:/Intern/Student_performance_data _.csv")

# Target and input features
y = df['GradeClass']
X = df.drop(['StudentID', 'GradeClass', 'GPA'], axis=1)

# Scale features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

# Save the model and scaler
joblib.dump(rf_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully!")
