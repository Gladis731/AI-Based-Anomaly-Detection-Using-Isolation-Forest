# AI-Based Anomaly Detection Using Isolation Forest
# Smart Classroom Attention Anomaly Detection

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Student engagement dataset
data = {
    "AttentionScore": [85, 78, 92, 88, 81, 76, 90, 87, 83, 79,
                       20, 25, 30, 18, 35, 82, 84, 91, 77, 86],
    "EyeFocus": [90, 82, 95, 85, 84, 80, 92, 89, 86, 81,
                 15, 20, 25, 12, 30, 88, 87, 94, 79, 90],
    "ActivityScore": [80, 75, 90, 82, 79, 73, 88, 84, 81, 77,
                      18, 22, 28, 15, 32, 80, 83, 89, 74, 85],
    "Attendance": [92, 88, 96, 91, 89, 85, 94, 90, 87, 86,
                   40, 35, 45, 30, 50, 90, 91, 95, 84, 92]
}

df = pd.DataFrame(data)

# Select features for anomaly detection
features = ["AttentionScore", "EyeFocus", "ActivityScore", "Attendance"]
X = df[features]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest model
model = IsolationForest(contamination=0.20, random_state=42)
df["Prediction"] = model.fit_predict(X_scaled)

# Convert prediction values into readable labels
df["Result"] = df["Prediction"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

# Display result table
print(df)

# Visualize anomaly detection result
normal = df[df["Result"] == "Normal"]
anomaly = df[df["Result"] == "Anomaly"]

plt.figure(figsize=(8, 5))
plt.scatter(normal.index, normal["AttentionScore"], label="Normal")
plt.scatter(anomaly.index, anomaly["AttentionScore"], label="Anomaly")

plt.title("Smart Classroom Attention Anomaly Detection")
plt.xlabel("Student Record")
plt.ylabel("Attention Score")
plt.legend()
plt.grid(True)
plt.show()
