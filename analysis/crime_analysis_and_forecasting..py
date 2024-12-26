import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from tabulate import tabulate


def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.

    """
    data = pd.read_csv(file_path).head(20000)
    data['Date Rptd'] = pd.to_datetime(data['Date Rptd'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    data = data[data['Date Rptd'].dt.year == 2020]

    # Filter crimes with at least 10 occurrences
    crime_counts = data['Crm Cd Desc'].value_counts()
    common_crimes = crime_counts[crime_counts >= 10].index
    data = data[data['Crm Cd Desc'].isin(common_crimes)]

    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].fillna("Unknown")
        else:
            data[column] = data[column].fillna(0)
    return data


def encode_and_split_data(data, selected_features, label_col):
    """
    Encode categorical features and split data into train and test sets.
    """
    encoder = LabelEncoder()
    for col in ['Vict Sex', 'Premis Desc', 'Weapon Desc']:
        if col in data.columns:
            data[col] = encoder.fit_transform(data[col])

    # Filter labels based on minimum occurrences
    label_counts = data[label_col].value_counts()
    valid_labels = label_counts[label_counts >= 5].index
    data = data[data[label_col].isin(valid_labels)]

    # Feature matrix and target vector
    X = data[selected_features]
    y = encoder.fit_transform(data[label_col])

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, encoder


def train_and_evaluate_model(X_train, X_test, y_train, y_test, encoder):
    """
    Train the Random Forest model and evaluate predictions.
    """
    scaler = StandardScaler()
    X_train[['Vict Age', 'LAT', 'LON']] = scaler.fit_transform(X_train[['Vict Age', 'LAT', 'LON']])
    X_test[['Vict Age', 'LAT', 'LON']] = scaler.transform(X_test[['Vict Age', 'LAT', 'LON']])

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    predicted_vs_actual = pd.DataFrame({
        'Actual': encoder.inverse_transform(y_test),
        'Predicted': encoder.inverse_transform(y_pred)
    })
    return predicted_vs_actual


def analyze_predictions(predicted_vs_actual):
    """
    Analyze prediction accuracy and visualize results.
    """
    crime_counts_actual = predicted_vs_actual['Actual'].value_counts().sort_index()
    crime_counts_predicted = predicted_vs_actual['Predicted'].value_counts().sort_index()

    crime_comparison = pd.DataFrame({
        'Actual': crime_counts_actual,
        'Predicted': crime_counts_predicted
    }).fillna(0)
    crime_comparison['Accuracy'] = crime_comparison['Actual'] / (
            crime_comparison['Actual'] + abs(crime_comparison['Actual'] - crime_comparison['Predicted'])
    )

    headers = ['Crime Type', 'Actual Count', 'Predicted Count', 'Accuracy']
    table = [
        [crime_type, int(row['Actual']), int(row['Predicted']), f"{row['Accuracy']:.2f}"]
        for crime_type, row in crime_comparison.iterrows()
    ]
    print(tabulate(table, headers=headers, tablefmt='fancy_grid', numalign="right", stralign="center"))

    top_10_accurate_crimes = crime_comparison.sort_values(by='Accuracy', ascending=False).head(10)
    top_10_accurate_crimes[['Actual', 'Predicted']].plot(kind='bar', figsize=(12, 6), color=['blue', 'orange'])
    plt.title('Top 10 Most Accurate Crimes - Actual vs Predicted')
    plt.xlabel('Crime Type')
    plt.ylabel('Number of Crimes')
    plt.xticks(rotation=45, ha='right')
    plt.legend(['Actual', 'Predicted'])
    plt.tight_layout()
    plt.show()


def forecast_crimes(data):
    """
    Predict future crime counts using ARIMA for the next 15 days.
    """
    crime_by_date = data.groupby(data['Date Rptd'].dt.date).size()
    model = ARIMA(crime_by_date, order=(5, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=15)
    forecast_dates = pd.date_range(start=crime_by_date.index[-1] + pd.Timedelta(days=1), periods=15, freq='D')

    predictions_table = [[date.date(), int(pred)] for date, pred in zip(forecast_dates, forecast)]
    print("\nForecasted Crimes for the Next 15 Days:")
    print(tabulate(predictions_table, headers=['Date', 'Predicted Crimes'], tablefmt='fancy_grid', numalign="right",
                   stralign="center"))

    plt.figure(figsize=(10, 6))
    plt.plot(crime_by_date.index, crime_by_date.values, label="Historical Data", color='blue')
    plt.plot(forecast_dates, forecast, label="Forecasted Data", color='red', linestyle='--')
    plt.title("Crime Prediction - Forecast for the Next 15 Days")
    plt.xlabel("Date")
    plt.ylabel("Number of Crimes")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filepath = ""
    df = load_and_preprocess_data(filepath)

    selected_feature = ['AREA', 'Vict Age', 'Vict Sex', 'Premis Desc', 'Weapon Desc', 'LAT', 'LON']
    label_column = 'Crm Cd Desc'

    X_train, X_test, y_train, y_test, encoder = encode_and_split_data(df, selected_feature, label_column)
    predicted_vs_real = train_and_evaluate_model(X_train, X_test, y_train, y_test, encoder)

    analyze_predictions(predicted_vs_real)
    forecast_crimes(df)
