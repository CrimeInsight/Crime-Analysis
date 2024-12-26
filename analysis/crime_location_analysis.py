import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import folium


def preprocess_data(file_path):
    """Preprocess the crime data with essential error handling."""
    try:
        if not file_path:
            raise ValueError("File path cannot be empty")

        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError("Dataset is empty")
        data = data.head(2000)

        data['Date Rptd'] = pd.to_datetime(data['Date Rptd'],
                                           format='%m/%d/%Y %I:%M:%S %p',
                                           errors='coerce')
        data['DATE OCC'] = pd.to_datetime(data['DATE OCC'],
                                          format='%m/%d/%Y %I:%M:%S %p',
                                          errors='coerce')

        if data['DATE OCC'].isna().all():
            raise ValueError("Failed to parse any dates in 'DATE OCC' column")

        # Time processing
        data['TIME OCC'] = data['TIME OCC'].astype(str).str.zfill(4)
        data['TIME OCC'] = pd.to_datetime(data['TIME OCC'],
                                          format='%H%M',
                                          errors='coerce').dt.time

        # Extracting time components
        data['hour'] = pd.to_datetime(data['TIME OCC'], errors='coerce').dt.hour
        data['day_of_week'] = data['DATE OCC'].dt.dayofweek
        data['month'] = data['DATE OCC'].dt.month

        # Handle missing hours safely
        hour_mean = data['hour'].mean()
        if pd.isna(hour_mean):
            hour_mean = 12  # Default to noon if all hours are missing
        data['hour'] = data['hour'].fillna(hour_mean)

        # Calculate cyclical time features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

        # Coordinate validation
        data = data.dropna(subset=['LAT', 'LON'])
        data = data[(data['LAT'] != 0) & (data['LON'] != 0)]

        if len(data) == 0:
            raise ValueError("No valid data points remaining after preprocessing")

        return data

    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or contains no valid data")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error preprocessing data: {str(e)}")


def actual_crime_situation(data):
    """Visualize current crime situation with error handling for critical operations."""
    try:
        required_columns = ['LAT', 'LON', 'AREA', 'DATE OCC', 'Crm Cd Desc']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {set(required_columns) - set(data.columns)}")

        data = data.dropna(subset=required_columns)
        if data.empty:
            raise ValueError("No valid data points for visualization")

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

        # Create map with validated coordinates
        mean_lat = data['LAT'].mean()
        mean_lon = data['LON'].mean()
        if not (-90 <= mean_lat <= 90) or not (-180 <= mean_lon <= 180):
            raise ValueError("Invalid coordinates for map center")

        heatmap_map_current = folium.Map(location=[mean_lat, mean_lon], zoom_start=10)

        for _, row in data.iterrows():
            if not (-90 <= row['LAT'] <= 90) or not (-180 <= row['LON'] <= 180):
                continue  # Skip invalid coordinates

            crime_type = row.get('Crm Cd Desc', 'Unknown')
            color = crime_type_colors.get(crime_type, default_color)
            location = [row['LAT'], row['LON']]

            folium.Circle(
                location=location,
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                tooltip=f"Crime: {crime_type}, Date: {row['DATE OCC']}, Location: {row['LAT']}, {row['LON']}"
            ).add_to(heatmap_map_current)

        try:
            heatmap_map_current.save("current_crime_heatmap.html")
            print("Current crime heatmap saved as current_crime_heatmap.html")
        except Exception as e:
            raise RuntimeError(f"Failed to save heatmap: {str(e)}")

    except Exception as e:
        raise RuntimeError(f"Error creating crime situation visualization: {str(e)}")


def perform_clustering(data, n_clusters=7):
    """Perform clustering with error handling for critical operations."""
    try:
        features = data[['LAT', 'LON', 'hour_sin', 'hour_cos']]
        if features.empty:
            raise ValueError("No features available for clustering")

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        features_imputed = imputer.fit_transform(features)

        if np.isnan(features_imputed).any():
            raise ValueError("Failed to impute missing values")

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data['cluster'] = kmeans.fit_predict(features_imputed)

        return data, kmeans.cluster_centers_

    except Exception as e:
        raise RuntimeError(f"Error performing clustering: {str(e)}")


def calculate_hotspot_proportions(data, cluster_centroids, threshold=0.1):
    """Calculate hotspot proportions with basic error handling."""
    try:
        if len(data) == 0:
            raise ValueError("Empty dataset")
        if len(cluster_centroids) == 0:
            raise ValueError("No cluster centroids provided")

        total_crimes = len(data)
        hotspot_proportions = []

        def calculate_distance(lat1, lon1, lat2, lon2):
            return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

        for cluster_id, centroid in enumerate(cluster_centroids):
            crimes_near_hotspot = data.apply(
                lambda row: calculate_distance(row['LAT'], row['LON'],
                                               centroid[0], centroid[1]) < threshold,
                axis=1
            ).sum()
            proportion = crimes_near_hotspot / total_crimes
            hotspot_proportions.append(proportion)

        return hotspot_proportions

    except Exception as e:
        raise RuntimeError(f"Error calculating hotspot proportions: {str(e)}")


def perform_supervised_learning(data):
    """Perform supervised learning with minimal error handling."""
    try:
        required_columns = ['hour_sin', 'hour_cos', 'day_of_week', 'month', 'cluster']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Missing required columns for supervised learning")

        X = data[['hour_sin', 'hour_cos', 'day_of_week', 'month']]
        y = data['cluster']

        if len(X) != len(y):
            raise ValueError("Feature and target dimensions do not match")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return model, X_test, y_test, y_pred

    except Exception as e:
        raise RuntimeError(f"Error in supervised learning: {str(e)}")


def compare_predictions_and_calculate_accuracy(data, y_test, y_pred):
    """Compare predictions with basic error handling."""
    try:
        if len(y_test) != len(y_pred):
            raise ValueError("Mismatched lengths between test and prediction arrays")

        if 'cluster' not in data.columns:
            raise ValueError("Cluster column missing from data")

        last_20_data = data.tail(int(0.2 * len(data)))
        actual_counts = last_20_data['cluster'].value_counts()
        predicted_counts = pd.Series(y_pred).value_counts()

        comparison = pd.DataFrame({
            'Predicted': predicted_counts,
            'Actual': actual_counts
        }).fillna(0).astype(int)

        print("\nCluster Prediction vs Actual Comparison:")
        print(comparison)

        total_correct_predictions = sum(
            min(comparison.loc[idx, 'Predicted'], comparison.loc[idx, 'Actual'])
            for idx in comparison.index
        )
        total_actual_values = comparison['Actual'].sum()

        if total_actual_values == 0:
            raise ValueError("No actual values available for comparison")

        overall_accuracy = (total_correct_predictions / total_actual_values) * 100
        print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")

        return overall_accuracy

    except Exception as e:
        raise RuntimeError(f"Error comparing predictions: {str(e)}")


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
    try:
        file_path = ""
        if not file_path:
            raise ValueError("Please provide a valid file path")

        data = preprocess_data(file_path)
        actual_crime_situation(data)

        data, cluster_centroids = perform_clustering(data, n_clusters=7)

        hotspot_proportions = calculate_hotspot_proportions(data, cluster_centroids,
                                                            threshold=0.1)
        for i, proportion in enumerate(hotspot_proportions):
            print(f"Cluster {i} Hotspot Proportion: {proportion * 100:.2f}%")

        model, X_test, y_test, y_pred = perform_supervised_learning(data)
        compare_predictions_and_calculate_accuracy(data, y_test, y_pred)

        visualize_clusters(data, cluster_centroids)

    except ValueError as e:
        print(f"Validation Error: {str(e)}")
    except FileNotFoundError as e:
        print(f"File Error: {str(e)}")
    except RuntimeError as e:
        print(f"Runtime Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")