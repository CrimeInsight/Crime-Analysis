import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from tabulate import tabulate


def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset with error handling.
    """
    try:
        if not file_path:
            raise ValueError("File path cannot be empty")

        data = pd.read_csv(file_path).head(20000)
        if data.empty:
            raise ValueError("Dataset is empty")

        data['Date Rptd'] = pd.to_datetime(data['Date Rptd'],
                                           format='%m/%d/%Y %I:%M:%S %p',
                                           errors='coerce')

        if data['Date Rptd'].isna().all():
            raise ValueError("Failed to parse any dates in 'Date Rptd' column")

        data = data[data['Date Rptd'].dt.year == 2020]
        if data.empty:
            raise ValueError("No data available for year 2020")

        crime_counts = data['Crm Cd Desc'].value_counts()
        common_crimes = crime_counts[crime_counts >= 10].index
        if len(common_crimes) == 0:
            raise ValueError("No crimes with 10 or more occurrences found")

        data = data[data['Crm Cd Desc'].isin(common_crimes)]

        for column in data.columns:
            if data[column].dtype == 'object':
                data[column] = data[column].fillna("Unknown")
            else:
                data[column] = data[column].fillna(0)

        return data

    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or contains no valid data")
    except pd.errors.ParserError:
        raise ValueError("Error parsing CSV file - please check the file format")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")


def encode_and_split_data(data, selected_features, label_col):
    """
    Encode categorical features and split data with error handling.
    """
    try:
        if not isinstance(selected_features, list) or not selected_features:
            raise ValueError("Selected features must be a non-empty list")

        if label_col not in data.columns:
            raise ValueError(f"Label column '{label_col}' not found in dataset")

        missing_features = [col for col in selected_features if col not in data.columns]
        if missing_features:
            raise ValueError(f"Features not found in dataset: {missing_features}")

        encoder = LabelEncoder()
        categorical_columns = ['Vict Sex', 'Premis Desc', 'Weapon Desc']

        for col in categorical_columns:
            if col in data.columns:
                try:
                    data[col] = encoder.fit_transform(data[col])
                except ValueError as e:
                    raise ValueError(f"Error encoding column {col}: {str(e)}")

        label_counts = data[label_col].value_counts()
        valid_labels = label_counts[label_counts >= 5].index
        if len(valid_labels) == 0:
            raise ValueError("No labels with 5 or more occurrences found")

        data = data[data[label_col].isin(valid_labels)]
        if data.empty:
            raise ValueError("No data remaining after filtering labels")

        X = data[selected_features]
        try:
            y = encoder.fit_transform(data[label_col])
        except ValueError as e:
            raise ValueError(f"Error encoding labels: {str(e)}")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test, encoder
        except ValueError as e:
            raise ValueError(f"Error splitting data: {str(e)}")

    except Exception as e:
        raise RuntimeError(f"Unexpected error in data encoding and splitting: {str(e)}")


def train_and_evaluate_model(X_train, X_test, y_train, y_test, encoder):
    """
    Train and evaluate model with error handling.
    """
    try:
        if X_train.empty or X_test.empty:
            raise ValueError("Training or test data is empty")

        if len(y_train) == 0 or len(y_test) == 0:
            raise ValueError("Training or test labels are empty")

        numeric_columns = ['Vict Age', 'LAT', 'LON']
        missing_columns = [col for col in numeric_columns if col not in X_train.columns]
        if missing_columns:
            raise ValueError(f"Required numeric columns missing: {missing_columns}")

        scaler = StandardScaler()
        try:
            X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
            X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
        except ValueError as e:
            raise ValueError(f"Error in data scaling: {str(e)}")

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        try:
            clf.fit(X_train, y_train)
        except ValueError as e:
            raise ValueError(f"Error training the model: {str(e)}")

        try:
            y_pred = clf.predict(X_test)
        except ValueError as e:
            raise ValueError(f"Error making predictions: {str(e)}")

        try:
            predicted_vs_actual = pd.DataFrame({
                'Actual': encoder.inverse_transform(y_test),
                'Predicted': encoder.inverse_transform(y_pred)
            })
            return predicted_vs_actual
        except ValueError as e:
            raise ValueError(f"Error creating results DataFrame: {str(e)}")

    except Exception as e:
        raise RuntimeError(f"Unexpected error in model training and evaluation: {str(e)}")


def analyze_predictions(predicted_vs_actual):
    """
    Analyze prediction accuracy and visualize results.
    """
    if predicted_vs_actual.empty:
        raise ValueError("No predictions to analyze")

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
    Forecast crimes with error handling.
    """
    try:
        if data.empty:
            raise ValueError("No data available for forecasting")

        if 'Date Rptd' not in data.columns:
            raise ValueError("Date column not found in dataset")

        crime_by_date = data.groupby(data['Date Rptd'].dt.date).size()
        if len(crime_by_date) < 6:
            raise ValueError("Insufficient data points for forecasting (minimum 6 required)")

        try:
            model = ARIMA(crime_by_date, order=(5, 1, 0))
            model_fit = model.fit()
        except Exception as e:
            raise ValueError(f"Error fitting ARIMA model: {str(e)}")

        try:
            forecast = model_fit.forecast(steps=15)
            forecast_dates = pd.date_range(
                start=crime_by_date.index[-1] + pd.Timedelta(days=1),
                periods=15,
                freq='D'
            )
        except Exception as e:
            raise ValueError(f"Error generating forecast: {str(e)}")

        predictions_table = [[date.date(), int(pred)] for date, pred in zip(forecast_dates, forecast)]
        print("\nForecasted Crimes for the Next 15 Days:")
        print(tabulate(predictions_table, headers=['Date', 'Predicted Crimes'],
                      tablefmt='fancy_grid', numalign="right", stralign="center"))

        try:
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
        except Exception as e:
            raise RuntimeError(f"Error creating forecast visualization: {str(e)}")

    except Exception as e:
        raise RuntimeError(f"Error in crime forecasting: {str(e)}")


if __name__ == "__main__":
    try:
        filepath = ""
        if not filepath:
            raise ValueError("Please provide a valid file path")

        df = load_and_preprocess_data(filepath)
        selected_feature = ['AREA', 'Vict Age', 'Vict Sex', 'Premis Desc',
                          'Weapon Desc', 'LAT', 'LON']
        label_column = 'Crm Cd Desc'

        X_train, X_test, y_train, y_test, encoder = encode_and_split_data(
            df, selected_feature, label_column
        )
        predicted_vs_real = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, encoder
        )

        analyze_predictions(predicted_vs_real)
        forecast_crimes(df)

    except ValueError as e:
        print(f"Validation Error: {str(e)}")
    except FileNotFoundError as e:
        print(f"File Error: {str(e)}")
    except RuntimeError as e:
        print(f"Runtime Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")