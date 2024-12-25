import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import folium
import matplotlib.pyplot as plt

# Load Data
data = pd.read_csv('Crime_Data_from_2020_to_Present.csv')

data = data.head(2000)

data['Date Rptd'] = pd.to_datetime(data['Date Rptd'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

# Convert 'DATE OCC' to datetime
data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

# Convert 'TIME OCC' to a proper time object (assumes HHMM format)
data['TIME OCC'] = data['TIME OCC'].astype(str).str.zfill(4)  # Ensure 4-digit format
data['TIME OCC'] = pd.to_datetime(data['TIME OCC'], format='%H%M', errors='coerce').dt.time

# Extract additional features
data['hour'] = pd.to_datetime(data['TIME OCC'], format='%H:%M:%S', errors='coerce').dt.hour  # Extract hour
data = data.dropna(subset=['LAT', 'LON'])  # Remove rows with missing latitude or longitude

# Feature Engineering
features = data[['LAT', 'LON', 'hour']]  # Select relevant features for clustering and prediction

# Clustering for Hotspots
dbscan = DBSCAN(eps=0.01, min_samples=10).fit(features[['LAT', 'LON']])
data['cluster'] = dbscan.labels_

# Visualize Hotspots
print("Generating hotspot map...")
m = folium.Map(location=[data['LAT'].mean(), data['LON'].mean()], zoom_start=12)
for _, row in data.iterrows():
    if row['cluster'] != -1:  # Ignore noise points
        folium.CircleMarker(
            location=(row['LAT'], row['LON']),
            radius=5,
            color='red' if row['cluster'] == 0 else 'blue',
            fill=True
        ).add_to(m)

# Save the hotspot map to an HTML file
m.save('hotspots.html')
print("Hotspot map saved as 'hotspots.html'.")

# Predictive Modeling
# Prepare data for training
X = features[['hour']]  # Example feature: hour of occurrence
y = features[['LAT', 'LON']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
print("Training prediction model...")
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict Future Locations
y_pred = model.predict(X_test)

# Visualize Predictions
print("Visualizing predictions...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test['LAT'], y_test['LON'], color='blue', label='Actual Locations', alpha=0.6)
plt.scatter(y_pred[:, 0], y_pred[:, 1], color='red', label='Predicted Locations', alpha=0.6)
plt.title("Actual vs Predicted Crime Locations")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.legend()
plt.grid(True)
plt.show()

print("All steps completed!")