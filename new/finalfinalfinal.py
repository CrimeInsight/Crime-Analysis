import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv('updated_dataset_with_uneven_crime_types.csv')

# Calculate crime-specific recommit probabilities based on the dataset
crime_recommit_probabilities = (
    df.groupby('Crime Type')['Return Status']
    .mean()  # Proportion of recommitting (1 = recommit, 0 = not recommit)
    .to_dict()
)

print("Calculated Recommit Probabilities by Crime Type:")
print(crime_recommit_probabilities)

# Add recommit probability column to the dataset
df['Recommit Probability'] = df['Crime Type'].map(crime_recommit_probabilities)

# Preprocessing: Encode categorical columns using LabelEncoder
label_encoder = LabelEncoder()

df['County of Indictment'] = label_encoder.fit_transform(df['County of Indictment'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Crime Type'] = label_encoder.fit_transform(df['Crime Type'])

# Features and target
X = df[['Release Year', 'County of Indictment', 'Gender', 'Age at Release', 'Crime Type']]
y = df['Return Status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities
probabilities = model.predict_proba(X_test)[:, 1]  # Probability of recommitting (class 1)

# Add predictions and probabilities to the test set
X_test['Predicted Probability'] = probabilities
X_test['Actual'] = y_test

# Decode the 'Crime Type' back to original labels for easier interpretation
X_test['Crime Type'] = df.loc[X_test.index, 'Crime Type']

# Analysis: Average predicted probability by crime type
average_probs = X_test.groupby('Crime Type')['Predicted Probability'].mean().sort_values(ascending=False)

# Plotting: Recommit Probability by Crime Type (Bar Plot)
plt.figure(figsize=(12, 6))
sns.barplot(x=average_probs.index, y=average_probs.values, palette='viridis')
plt.xticks(rotation=45)
plt.title('Average Predicted Probability of Recommitting by Crime Type')
plt.xlabel('Crime Type')
plt.ylabel('Average Probability of Recommitting')
plt.show()

# Save updated dataset with probabilities
df.to_csv('final_dataset_with_calculated_probabilities.csv', index=False)
print("Dataset saved with calculated probabilities!")
