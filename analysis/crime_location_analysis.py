import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import folium


def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.head(2000)

    data['Date Rptd'] = pd.to_datetime(data['Date Rptd'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    data['TIME OCC'] = data['TIME OCC'].astype(str).str.zfill(4)
    data['TIME OCC'] = pd.to_datetime(data['TIME OCC'], format='%H%M', errors='coerce').dt.time

    data['hour'] = pd.to_datetime(data['TIME OCC'], errors='coerce').dt.hour
    data['day_of_week'] = data['DATE OCC'].dt.dayofweek
    data['month'] = data['DATE OCC'].dt.month

    data['hour'] = data['hour'].fillna(data['hour'].mean())

    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

    data = data.dropna(subset=['LAT', 'LON'])
    data = data[(data['LAT'] != 0) & (data['LON'] != 0)]

    return data


def actual_crime_situation(data):
    data = data.dropna(subset=['LAT', 'LON', 'AREA', 'DATE OCC', 'Crm Cd Desc'])

    data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p')

    # Define crime type colors
    crime_type_colors = {
        "ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT": "red",
        "BURGLARY": "blue",
        "ROBBERY": "purple",
        "VEHICLE - STOLEN": "green",
        "SHOPLIFTING-GRAND THEFT ($950.01 & OVER)": "green",
        "VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)": "orange"
    }
    default_color = "gray"

    heatmap_map_current = folium.Map(location=[data['LAT'].mean(), data['LON'].mean()], zoom_start=10)

    for _, row in data.iterrows():
        crime_type = row.get('Crm Cd Desc', 'Unknown')
        color = crime_type_colors.get(crime_type, default_color)

        # Get the crime location
        location = [row['LAT'], row['LON']]

        radius = 5

        folium.Circle(
            location=location,
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            tooltip=f"Crime: {crime_type}, Date: {row['DATE OCC']}, Location: {row['LAT']}, {row['LON']}"
        ).add_to(heatmap_map_current)

    heatmap_map_current.save("current_crime_heatmap.html")
    print("Current crime heatmap saved as current_crime_heatmap.html")


# Perform K-Means clustering on the data
def perform_clustering(data, n_clusters=7):
    features = data[['LAT', 'LON', 'hour_sin', 'hour_cos']]

    # Handle missing values in features
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(features_imputed)

    return data, kmeans.cluster_centers_


# Calculate the proportion of crimes near each cluster centroid (hotspot)
def calculate_hotspot_proportions(data, cluster_centroids, threshold=0.1):
    total_crimes = len(data)

    def calculate_distance(lat1, lon1, lat2, lon2):
        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

    hotspot_proportions = []
    for cluster_id, centroid in enumerate(cluster_centroids):
        crimes_near_hotspot = data.apply(
            lambda row: calculate_distance(row['LAT'], row['LON'], centroid[0], centroid[1]) < threshold, axis=1
        ).sum()
        proportion = crimes_near_hotspot / total_crimes
        hotspot_proportions.append(proportion)

    return hotspot_proportions


# Train a Random Forest classifier and predict crime clusters
def perform_supervised_learning(data):
    X = data[['hour_sin', 'hour_cos', 'day_of_week', 'month']]
    y = data['cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred


# Compare predictions to actual clusters and calculate accuracy
def compare_predictions_and_calculate_accuracy(data, y_test, y_pred):
    last_20_data = data.tail(int(0.2 * len(data)))
    actual_counts = last_20_data['cluster'].value_counts()

    predicted_counts = pd.Series(y_pred).value_counts()

    comparison = pd.DataFrame({'Predicted': predicted_counts, 'Actual': actual_counts}).fillna(0).astype(int)
    print("\nCluster Prediction vs Actual Comparison:")
    print(comparison)

    total_correct_predictions = sum(min(comparison.loc[idx, 'Predicted'], comparison.loc[idx, 'Actual'])
                                    for idx in comparison.index)
    total_actual_values = comparison['Actual'].sum()

    overall_accuracy = (total_correct_predictions / total_actual_values) * 100
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")

    return overall_accuracy


# Visualize crime clusters on a map
def visualize_clusters(data, cluster_centroids):
    m = folium.Map(location=[data['LAT'].mean(), data['LON'].mean()], zoom_start=12)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'brown']

    for _, row in data.iterrows():
        folium.CircleMarker(
            location=(row['LAT'], row['LON']),
            radius=5,
            color=colors[row['cluster'] % len(colors)],
            fill=True
        ).add_to(m)

    for cluster, centroid in enumerate(cluster_centroids):
        folium.Marker(
            location=(centroid[0], centroid[1]),
            popup=f'Cluster {cluster}',
            icon=folium.Icon(color='black', icon='info-sign')
        ).add_to(m)

    m.save('hotspots.html')
    print("Map saved as 'hotspots.html'")


if __name__ == "__main__":
    file_path = ""

    data = preprocess_data(file_path)

    # display current situation
    actual_crime_situation(data)

    # Perform clustering
    data, cluster_centroids = perform_clustering(data, n_clusters=7)

    # Calculate hotspot proportions
    hotspot_proportions = calculate_hotspot_proportions(data, cluster_centroids, threshold=0.1)
    for i, proportion in enumerate(hotspot_proportions):
        print(f"Cluster {i} Hotspot Proportion: {proportion * 100:.2f}%")

    # Train and evaluate a supervised model
    model, X_test, y_test, y_pred = perform_supervised_learning(data)
    compare_predictions_and_calculate_accuracy(data, y_test, y_pred)

    # Visualize clusters on a map
    visualize_clusters(data, cluster_centroids)
