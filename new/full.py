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

# Gender Distribution
gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(6, 6))
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='viridis')
plt.title('Gender Distribution')
plt.ylabel('Count')
plt.xlabel('Gender')
plt.show()


# Correlation Heatmap

# Create a copy for correlation analysis
df_corr = df.copy()

# Encode categorical columns for correlation
label_encoder = LabelEncoder()
df_corr['Gender'] = label_encoder.fit_transform(df_corr['Gender'])
df_corr['County of Indictment'] = label_encoder.fit_transform(df_corr['County of Indictment'])
df_corr['Crime Type'] = label_encoder.fit_transform(df_corr['Crime Type'])
df_corr['Return Status'] = label_encoder.fit_transform(df_corr['Return Status'])

# Create correlation heatmap with all relevant columns
plt.figure(figsize=(12, 10))
correlation_matrix = df_corr[['Release Year', 'County of Indictment', 'Gender', 
                            'Age at Release', 'Return Status', 'Crime Type']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (All Features)')
plt.tight_layout()
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


# --- Model Comparison Plot ---
model_accuracies = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [logistic_accuracy, rf_accuracy]
})

plt.figure(figsize=(8, 6))
sns.barplot(x='Model', y='Accuracy', data=model_accuracies, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Accuracy is between 0 and 1
plt.show()



# --- Code Quality ---
# # Save requirements
# with open('requirements.txt', 'w') as f:
#     f.write("pandas\nnumpy\nmatplotlib\nseaborn\nscikit-learn\n")
