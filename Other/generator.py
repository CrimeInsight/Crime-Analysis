import pandas as pd
import random
import numpy as np

# Define the columns and possible values
ages = list(range(18, 65))  # Age at release between 18 and 64
genders = ['MALE', 'FEMALE']
previous_offenses = list(range(0, 6))  # Number of previous offenses (0 to 5)
crime_types = ['Theft', 'Drug Offense', 'Violent Crime', 'Fraud', 'Other']
substance_abuse = ['Yes', 'No']
mental_health_status = ['Yes', 'No']
employment_status = ['Employed', 'Unemployed']
return_status = ['Returned Parole Violation', 'Not Returned']

# Generate random data
data = {
    'Age at Release': [random.choice(ages) for _ in range(500)],
    'Gender': [random.choice(genders) for _ in range(500)],
    'Previous Offenses': [random.choice(previous_offenses) for _ in range(500)],
    'Type of Crime': [random.choice(crime_types) for _ in range(500)],
    'Substance Abuse History': [random.choice(substance_abuse) for _ in range(500)],
    'Mental Health Status': [random.choice(mental_health_status) for _ in range(500)],
    'Employment Status': [random.choice(employment_status) for _ in range(500)],
    'Return Status': [random.choice(return_status) for _ in range(500)],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows
print(df.head())

# Optionally, save to CSV
df.to_csv("recidivism_data.csv", index=False)
