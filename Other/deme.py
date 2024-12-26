import pandas as pd
import random

# Load the dataset
df = pd.read_csv('2008.csv')

# List of crime types
crime_types = [
    'Narco Dealer', 'Murder', 'Theft', 'Assault', 'Fraud', 
    'Robbery', 'Rape', 'Burglary', 'Money Laundering', 
    'Kidnapping', 'Arson', 'Corruption', 'Drug Possession', 
    'Sexual Offense', 'Embezzlement'
]

# Define the weighted probabilities for each crime type
crime_probabilities = {
    'Narco Dealer': 0.10,
    'Murder': 0.05,
    'Theft': 0.20,
    'Assault': 0.10,
    'Fraud': 0.15,
    'Robbery': 0.05,
    'Rape': 0.05,
    'Burglary': 0.08,
    'Money Laundering': 0.03,
    'Kidnapping': 0.04,
    'Arson': 0.03,
    'Corruption': 0.02,
    'Drug Possession': 0.05,
    'Sexual Offense': 0.02,
    'Embezzlement': 0.01
}

# Normalize the probabilities to sum to 1
total_prob = sum(crime_probabilities.values())
normalized_probabilities = {k: v / total_prob for k, v in crime_probabilities.items()}

# Function to assign a random crime type with uneven distribution
def assign_random_crime_type():
    return random.choices(crime_types, weights=[normalized_probabilities[crime] for crime in crime_types])[0]

# Apply the function to create the new column
df['Crime Type'] = df.apply(lambda row: assign_random_crime_type(), axis=1)

# Save the updated dataset to a new CSV file
df.to_csv('updated_dataset_with_uneven_crime_types.csv', index=False)

print("Crime Type column added with unevenly distributed crime types!")
