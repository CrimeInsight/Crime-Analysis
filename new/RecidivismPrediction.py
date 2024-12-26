import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Loading ---
# Load dataset
df = pd.read_csv('updated_dataset_with_uneven_crime_types.csv')

# --- Data Cleaning ---
# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

# Handle missing values
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())  # Fill numeric NaNs with median
df[categorical_cols] = df[categorical_cols].fillna('Unknown')  # Fill categorical NaNs with 'Unknown'

# Handle outliers using IQR (for numeric columns)
for col in ['Age at Release', 'Release Year']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# --- Data Preprocessing ---
# Encode categorical columns
label_encoder = LabelEncoder()
for col in ['County of Indictment', 'Gender', 'Crime Type']:
    df[col] = label_encoder.fit_transform(df[col])

# Standardize numeric features
scaler = StandardScaler()
df[['Age at Release', 'Release Year']] = scaler.fit_transform(df[['Age at Release', 'Release Year']])

# --- Feature Selection and Target ---
X = df[['Release Year', 'County of Indictment', 'Gender', 'Age at Release', 'Crime Type']]
y = df['Return Status']

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Machine Learning Model ---
# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Predict probabilities
probabilities = model.predict_proba(X_test)[:, 1]  # Probability of recommitting (class 1)

# --- Visualization ---
# Add predictions and probabilities to the test set
X_test = X_test.copy()
X_test['Predicted Probability'] = probabilities
X_test['Actual'] = y_test

# Decode the 'Crime Type' back to original labels for easier interpretation
df['Crime Type'] = label_encoder.inverse_transform(df['Crime Type'])
X_test['Crime Type'] = df.loc[X_test.index, 'Crime Type']

# Plotting: Recommit Probability by Crime Type
plt.figure(figsize=(12, 6))
sns.barplot(x='Crime Type', y='Predicted Probability', data=X_test, estimator='mean', palette='viridis')
plt.xticks(rotation=45)
plt.title('Probability of Recommitting by Crime Type')
plt.xlabel('Crime Type')
plt.ylabel('Average Probability of Recommitting')
plt.show()
