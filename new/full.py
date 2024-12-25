import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# --- Data Loading and Cleaning ---
# Load dataset
df = pd.read_csv('updated_dataset_with_uneven_crime_types.csv')

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric NaNs with median
df.fillna('Unknown', inplace=True)  # Fill categorical NaNs with 'Unknown'

# Handle outliers using IQR
for col in ['Age at Release', 'Release Year']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# --- Exploratory Data Analysis ---
# Crime Type Distribution
crime_type_counts = df['Crime Type'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(crime_type_counts, labels=crime_type_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Crime Type Distribution')
plt.show()

# Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Age at Release'], kde=True, color='blue', bins=20)
plt.title('Age Distribution')
plt.xlabel('Age at Release')
plt.ylabel('Frequency')
plt.show()

# Release Year vs Crime Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Crime Type', y='Release Year', data=df, palette='Set2')
plt.xticks(rotation=45)
plt.title('Release Year vs Crime Type')
plt.show()

# Gender Distribution
gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(6, 6))
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='viridis')
plt.title('Gender Distribution')
plt.ylabel('Count')
plt.xlabel('Gender')
plt.show()


# Correlation Heatmap

plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# --- Data Preprocessing ---
# Encode categorical columns
label_encoder = LabelEncoder()
for col in ['County of Indictment', 'Gender', 'Crime Type']:
    df[col] = label_encoder.fit_transform(df[col])

# Standardize features
scaler = StandardScaler()
df[['Age at Release', 'Release Year']] = scaler.fit_transform(df[['Age at Release', 'Release Year']])

# Features and target
X = df[['Release Year', 'County of Indictment', 'Gender', 'Age at Release', 'Crime Type']]
y = df['Return Status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)





# --- Machine Learning Models ---
# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# Model Comparison
print("Logistic Regression Accuracy:", logistic_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

# Classification Report and Confusion Matrix
print("\nClassification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_logistic))
print("\nConfusion Matrix for Logistic Regression:")
print(confusion_matrix(y_test, y_pred_logistic))

print("\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix for Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))

# --- Feature Importance (Random Forest) ---
feature_importances = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances['Importance'], y=feature_importances.index, palette='cool')
plt.title('Feature Importance (Random Forest)')
plt.show()



# --- Probability of Recommitting by Crime Type --- 

# Predict probabilities for Logistic Regression
proba_logistic = logistic_model.predict_proba(X_test)[:, 1]  # Probability of reoffending
X_test['Crime Type'] = X_test['Crime Type']  # Ensure Crime Type is in the test set
X_test['Logistic_Probability'] = proba_logistic

# Predict probabilities for Random Forest
proba_rf = rf_model.predict_proba(X_test[['Release Year', 'County of Indictment', 'Gender', 'Age at Release', 'Crime Type']])[:, 1]
X_test['RF_Probability'] = proba_rf

# Aggregate probabilities by Crime Type
crime_type_probs = X_test.groupby('Crime Type')[['Logistic_Probability', 'RF_Probability']].mean()

# Plot comparison
plt.figure(figsize=(10, 6))
crime_type_probs.plot(kind='bar', figsize=(10, 6), color=['blue', 'green'], alpha=0.7)
plt.title('Probability of Recommitting by Crime Type (Logistic Regression vs Random Forest)')
plt.ylabel('Average Probability')
plt.xlabel('Crime Type')
plt.xticks(rotation=45, ha='right')
plt.legend(['Logistic Regression', 'Random Forest'])
plt.tight_layout()
plt.show()

# --- Evaluation ---
# Compare accuracies
print("Logistic Regression Accuracy:", logistic_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

# Display probabilities
print("\nAverage Probability of Recommitting by Crime Type:")
print(crime_type_probs)


# --- Code Quality ---
# # Save requirements
# with open('requirements.txt', 'w') as f:
#     f.write("pandas\nnumpy\nmatplotlib\nseaborn\nscikit-learn\n")
