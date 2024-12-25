import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv("Crime_Data_from_2020_to_Present.csv")

crime_counts = data['Crm Cd Desc'].value_counts()
min_occurrences = 10
common_crimes = crime_counts[crime_counts >= min_occurrences].index
data = data[data['Crm Cd Desc'].isin(common_crimes)]

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].fillna("Unknown")
    else:
        data[column] = data[column].fillna(0)  # Replace missing values with

selected_features = [
    'AREA', 'Vict Age', 'Vict Sex', 'Premis Desc', 'Weapon Desc', 'LAT', 'LON'
]
label_col = 'Crm Cd Desc'

encoder = LabelEncoder()
for col in ['Vict Sex', 'Premis Desc', 'Weapon Desc']:
    if col in data.columns:  # Ensure the column exists in the dataset
        data[col] = encoder.fit_transform(data[col])

label_counts = data[label_col].value_counts()
min_label_occurrences = 5
valid_labels = label_counts[label_counts >= min_label_occurrences].index
data = data[data[label_col].isin(valid_labels)]

X = data[selected_features]
y = encoder.fit_transform(data[label_col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train[['Vict Age', 'LAT', 'LON']] = scaler.fit_transform(X_train[['Vict Age', 'LAT', 'LON']])
X_test[['Vict Age', 'LAT', 'LON']] = scaler.transform(X_test[['Vict Age', 'LAT', 'LON']])

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Future
data['Date Rptd'] = pd.to_datetime(data['Date Rptd'], errors='coerce')

crime_by_date = data.groupby(data['Date Rptd'].dt.date).size()

train = crime_by_date[:-30]  # Use all but the last 30 days for training
test = crime_by_date[-30:]   # Use the last 30 days for testing

model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Make predictions for the next 30 days
forecast = model_fit.forecast(steps=30)

plt.figure(figsize=(10, 6))
plt.plot(crime_by_date.index, crime_by_date.values, label="Historical Data")
plt.plot(pd.date_range(start=crime_by_date.index[-1], periods=31, freq='D')[1:], forecast, label="Forecast", color='red')
plt.title("Crime Prediction - Forecast for the Next 30 Days")
plt.xlabel("Date")
plt.ylabel("Number of Crimes")
plt.legend()
plt.show()
