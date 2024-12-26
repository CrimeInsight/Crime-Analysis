import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('updated_dataset_with_uneven_crime_types.csv')

# Create a pie chart showing the distribution of crime types
crime_type_counts = df['Crime Type'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(crime_type_counts, labels=crime_type_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Crime Type Distribution')
plt.show()


# ---c

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv('updated_dataset_with_uneven_crime_types.csv')

# Preprocessing: Encode categorical columns using LabelEncoder
label_encoder = LabelEncoder()

df['County of Indictment'] = label_encoder.fit_transform(df['County of Indictment'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Return Status'] = label_encoder.fit_transform(df['Return Status'])
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
df['Crime Type'] = label_encoder.inverse_transform(df['Crime Type'])

# Merge with the original Crime Type
X_test['Crime Type'] = df.loc[X_test.index, 'Crime Type']

# Plotting: Recommit Probability by Crime Type (Bar Plot or Violin Plot)
plt.figure(figsize=(12, 6))

# You can use either a bar plot or a violin plot
sns.barplot(x='Crime Type', y='Predicted Probability', data=X_test, estimator='mean', palette='viridis')
# Or use a violin plot for distribution
# sns.violinplot(x='Crime Type', y='Predicted Probability', data=X_test, palette='viridis')

plt.xticks(rotation=45)
plt.title('Probability of Recommitting by Crime Type')
plt.xlabel('Crime Type')
plt.ylabel('Average Probability of Recommitting')
plt.show()
