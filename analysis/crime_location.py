from analysis import Analysis

class CrimeLocationAnalysis(Analysis):
    def __init__(self):
        super().__init__()
        self.data = None

    def clean(self):
        # Load and clean the dataset
        self.data = self.pd.read_csv(self.path)
        self.data = self.data.dropna(subset=['LAT', 'LON', 'AREA', 'DATE OCC', 'Crm Cd Desc'])
        self.data['DATE OCC'] = self.pd.to_datetime(self.data['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p')
        self.data = self.data[(self.data['LAT'] != 0) & (self.data['LON'] != 0)]
        print("Data cleaned")
        return self.data

    def plot(self):
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

        # Create a heatmap
        heatmap_map = self.folium.Map(location=[self.data['LAT'].mean(), self.data['LON'].mean()], zoom_start=10)
        for _, row in self.data.iterrows():
            crime_type = row.get('Crm Cd Desc', 'Unknown')
            color = crime_type_colors.get(crime_type, default_color)
            location = [row['LAT'], row['LON']]
            self.folium.Circle(
                location=location,
                radius=5,  # Fixed size for the circle
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                tooltip=f"Crime: {crime_type}, Date: {row['DATE OCC']}, Location: {row['LAT']}, {row['LON']}"
            ).add_to(heatmap_map)

        # Save the map
        heatmap_map.save("output/current_crime_heatmap.html")
        print("Current crime heatmap saved as current_crime_heatmap.html")

    def train(self):
        # Placeholder for future functionality
        print("Training logic not implemented yet")
