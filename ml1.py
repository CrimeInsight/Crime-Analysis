import pandas as pd
import folium

df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")

df = df.dropna(subset=['LAT', 'LON', 'AREA', 'DATE OCC', 'Crm Cd Desc'])

df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p')

# Remove rows with LAT or LON equal to 0 (invalid coordinates)
df = df[(df['LAT'] != 0) & (df['LON'] != 0)]

# Define crime type colors
crime_type_colors = {
    "ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT": "red",
    "BURGLARY": "blue",
    "ROBBERY": "purple",
    "VEHICLE - STOLEN": "green",
    "SHOPLIFTING-GRAND THEFT ($950.01 & OVER)": "green",
    "VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)": "orange"
}
# Default color for other crime types
default_color = "gray"

heatmap_map_current = folium.Map(location=[df['LAT'].mean(), df['LON'].mean()], zoom_start=10)

for _, row in df.iterrows():
    crime_type = row.get('Crm Cd Desc', 'Unknown')
    color = crime_type_colors.get(crime_type, default_color)  # Get color based on crime type

    # Get the crime location
    location = [row['LAT'], row['LON']]

    # Set a fixed size for the circle based on crime count
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
