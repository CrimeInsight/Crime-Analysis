import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("")
data = data.head(1000)
data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

data['Year'] = data['DATE OCC'].dt.year
data['Month'] = data['DATE OCC'].dt.month
data['Day of Week'] = data['DATE OCC'].dt.day_name()

data['TIME OCC'] = data['TIME OCC'].astype(str).str.zfill(4)
data['Hour'] = data['TIME OCC'].str[:2].astype(int)


# --- Crime Trend Over Time ---
def crime_trend_over_time():
    crime_trend = data.groupby('DATE OCC').size()

    crime_trend_smooth = crime_trend.rolling(window=7).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(crime_trend.index, crime_trend.values, color='blue', linewidth=2, label='Daily Crimes')
    plt.plot(crime_trend_smooth.index, crime_trend_smooth.values, color='orange', linestyle='--', linewidth=2, label='7-Day Moving Average')

    plt.scatter(crime_trend.index, crime_trend.values, color='blue', s=20, alpha=0.5)

    # Titles and Labels
    plt.title('Crime Trend Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Crimes', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')

    plt.grid(True)

    plt.tight_layout()
    plt.show()


# --- Heatmap for Hour and Day of the Week ---
def heatmap_for_hour_and_day():
    heatmap_data = data.groupby(['Hour', 'Day of Week']).size().unstack(fill_value=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d', cbar=True, linewidths=0.5, linecolor='gray', annot_kws={'size': 10, 'weight': 'bold'})

    plt.title('Crimes by Hour and Day of the Week', fontsize=16, fontweight='bold')
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Hour of the Day', fontsize=12)

    plt.tight_layout()
    plt.show()


# --- Crime Frequency by Year ---
def crime_frequency_by_year():
    crime_by_year = data['Year'].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    bars = crime_by_year.plot(kind='bar', color='skyblue', edgecolor='black', width=0.7)

    plt.title('Number of Crimes by Year', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Crimes', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)

    for bar in bars.patches:
        bars.annotate(f'{bar.get_height():,.0f}',
                      (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                      ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# --- Crime Frequency by Month ---
def crime_frequency_by_month():
    crime_by_month = data.groupby('Month').size()

    plt.figure(figsize=(8, 5))
    bars = crime_by_month.plot(kind='bar', color='coral', edgecolor='black', width=0.7)

    plt.title('Number of Crimes by Month', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Crimes', fontsize=12)

    for bar in bars.patches:
        bars.annotate(f'{bar.get_height():,.0f}',
                      (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                      ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

    plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


# --- Age Distribution ---
def age_distribution():
    valid_ages = data[data['Vict Age'] > 0]

    plt.figure(figsize=(12, 6))
    sns.histplot(valid_ages['Vict Age'], bins=30, kde=True, color='skyblue', stat='density', linewidth=2)

    sns.rugplot(valid_ages['Vict Age'], color='red', height=0.1)

    plt.title('Age Distribution of Victims', fontsize=18)
    plt.xlabel('Victim Age', fontsize=14)
    plt.ylabel('Density', fontsize=14)

    plt.grid(True)
    plt.show()


# --- Gender Analysis ---
def gender_analysis():
    gender_counts = data['Vict Sex'].dropna().value_counts()

    colors = ['#66b3ff', '#ffb3e6', '#cccccc']
    explode = (0.1, 0.1, 0)

    # Create the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode, wedgeprops={'edgecolor': 'black', 'linewidth': 1})

    plt.title('Victim Gender Distribution', fontsize=20, fontweight='bold')

    plt.legend(gender_counts.index, title='Gender', loc='upper left', fontsize=12)

    plt.axis('equal')
    plt.show()


# --- Descent Distribution ---
def descent_distribution():
    required_columns = ['Vict Descent', 'Crm Cd Desc', 'Crm Cd 1', 'Crm Cd 2',
                        'Crm Cd 3', 'Crm Cd 4', 'Weapon Desc', 'Premis Desc']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing columns: {', '.join(missing_columns)}")

    valid_descents = data['Vict Descent'].dropna()
    data.fillna({'Crm Cd Desc': 'Unknown', 'Weapon Desc': 'Unknown', 'Premis Desc': 'Unknown'}, inplace=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))
    ax = sns.countplot(x=valid_descents, palette="Set2", edgecolor="black", hue=valid_descents, dodge=False,
                       legend=False)

    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 10),
                    textcoords='offset points')

    plt.title('Victim Descent Distribution', fontsize=18)
    plt.xlabel('Victim Descent', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.show()


# --- Crime Types: Most Common Crimes ---
def crime_types():
    plt.figure(figsize=(16, 8))

    common_crimes = data['Crm Cd Desc'].value_counts().head(10)

    sns.barplot(
        x=common_crimes.values,
        y=common_crimes.index,
        hue=common_crimes.index,
        palette="coolwarm",
        dodge=False,
        legend=False
    )

    plt.title('Top 10 Most Common Crimes', fontsize=16)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Crime Description', fontsize=12)

    plt.xlim(0, common_crimes.values.max() + 100)
    for i, v in enumerate(common_crimes.values):
        plt.text(v + 20, i, str(v), va='center', fontsize=10)

    plt.subplots_adjust(left=0.3)
    plt.tight_layout()
    plt.show()

    # --- Multi-Class Crimes ---
    multi_class_crimes = pd.concat([
        data['Crm Cd 1'],
        data['Crm Cd 2'],
        data['Crm Cd 3'],
        data['Crm Cd 4']
    ]).dropna()

    multi_class_crime_counts = multi_class_crimes.value_counts().head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=multi_class_crime_counts.values,
        y=multi_class_crime_counts.index,
        hue=multi_class_crime_counts.index,
        dodge=False,
        palette="viridis",
        legend=False
    )
    plt.title('Top 10 Multi-Class Crimes', fontsize=16)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Crime Codes', fontsize=12)

    for i, v in enumerate(multi_class_crime_counts.values):
        plt.text(v + 50, i, str(v), va='center', fontsize=10)
    plt.show()


# --- Weapon Analysis ---
def weapon_analysis():
    plt.figure(figsize=(16, 8))

    common_weapons = data['Weapon Desc'].value_counts().head(10)

    sns.barplot(
        x=common_weapons.values,
        y=common_weapons.index,
        hue=common_weapons.index,
        palette="magma",
        dodge=False,
        legend=False
    )

    plt.title('Top 10 Most Commonly Used Weapons', fontsize=16)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Weapon Description', fontsize=12)

    plt.xlim(0, common_weapons.values.max() + 50)
    for i, v in enumerate(common_weapons.values):
        plt.text(v + 10, i, str(v), va='center', fontsize=10)

    plt.subplots_adjust(left=0.3)
    plt.tight_layout()
    plt.show()


# --- Premises Analysis ---
def premises_analysis():
    plt.figure(figsize=(16, 8))
    common_premises = data['Premis Desc'].value_counts().head(10)
    sns.barplot(
        x=common_premises.values,
        y=common_premises.index,
        hue=common_premises.index,
        palette="plasma",
        dodge=False,
        legend=False
    )
    plt.title('Top 10 Crime Premises', fontsize=16)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Premises Description', fontsize=12)
    plt.xlim(0, common_premises.values.max() + 50)
    for i, v in enumerate(common_premises.values):
        plt.text(v + 20, i, str(v), va='center', fontsize=10)
    plt.subplots_adjust(left=0.3)
    plt.tight_layout()
    plt.show()


# --- Map Status Codes to Meaningful Categories ---
def status_analysis():
    status_mapping = {
        'AA': 'Closed',    # Adult Arrest (Resolved)
        'IC': 'Open',      # Investigation Continued (Open)
        'AO': 'Closed',    # Adult Other (Resolved)
    }

    data['Status'] = data['Status'].map(status_mapping)

    print("Unique Status Values after mapping: ", data['Status'].unique())

    # --- Case Status ---
    status_counts = data['Status'].dropna().value_counts()

    explode = [0.1 if i == 0 else 0 for i in range(len(status_counts))]

    plt.figure(figsize=(8, 8))
    status_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'],
                       explode=explode, wedgeprops={'edgecolor': 'black', 'linewidth': 1})

    plt.title('Case Status Distribution', fontsize=18, fontweight='bold')
    plt.ylabel('')

    plt.legend(status_counts.index, title='Status', loc='best', fontsize=12)

    plt.axis('equal')
    plt.show()

    # --- Resolution Rates ---
    resolution_data = data.groupby(['Crm Cd Desc', 'Status']).size().unstack(fill_value=0)

    if 'Closed' in resolution_data.columns and 'Open' in resolution_data.columns:
        resolution_data['Resolution Rate'] = resolution_data['Closed'] / (
                    resolution_data['Open'] + resolution_data['Closed'])
    else:
        print("Missing 'Closed' or 'Open' statuses. Please check the 'Status' column.")

    if 'Resolution Rate' in resolution_data.columns:
        resolution_data.reset_index(inplace=True)
        resolution_data = resolution_data.dropna(subset=['Resolution Rate'])

        # Split data into chunks of 10 crime types
        chunk_size = 10
        num_chunks = (len(resolution_data) // chunk_size) + (1 if len(resolution_data) % chunk_size > 0 else 0)

        for i in range(num_chunks):
            chunk = resolution_data[i * chunk_size:(i + 1) * chunk_size]

            plt.figure(figsize=(12, 8))
            sns.barplot(x='Crm Cd Desc', y='Resolution Rate', data=chunk, hue='Crm Cd Desc', palette='viridis',
                        dodge=False, legend=False)

            plt.title(f'Crime Resolution Rates by Crime Type (Batch {i + 1})', fontsize=18, fontweight='bold')
            plt.xlabel('Crime Type', fontsize=14)
            plt.ylabel('Resolution Rate (Closed / Total)', fontsize=14)

            plt.xticks(rotation=45, ha='right')

            for p in plt.gca().patches:
                plt.gca().annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                   ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                                   textcoords='offset points')

            plt.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.show()
    else:
        print("No 'Resolution Rate' calculated. Ensure 'Open' and 'Closed' statuses are present in the data.")
