import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# start by loading the dataset
def load(nrows):
    return pd.read_csv("Crime_Data_from_2020_to_Present.csv", nrows=nrows)

# filter the dataset to contain only the necessary columns
def filter(data):
    # the necessary columns for my part of the project
    columns = ['Crm Cd Desc', 'Vict Sex', 'Premis Desc', 'TIME OCC', 'AREA', 'Vict Age', 'Weapon Used Cd']

    # filter the dataset to include only these columns
    return data[columns].copy()

def handle_missing_values(data):
    # first check for missing values
    print("Missing values per column:\n", data.isnull().sum())

    # we see that a lot of rows have missing values, like victim sex, premise description, and weapon used code.
    # this is because some of the crimes did not involve a victim, weapon, or did not occur at a premise
    # because of this i fill these null values with a string representing that the value is not applicable
    data['Vict Sex'] = data['Vict Sex'].fillna("Not Applicable")
    data['Weapon Used Cd'] = data['Weapon Used Cd'].fillna(0)
    data['Premis Desc'] = data['Premis Desc'].fillna("Unknown Premises")

    # lets re-check the missing values
    print("Missing values per column after adjustments:\n", data.isnull().sum())

def feature_engineering(data):
    # next we transform the columns in the dataset to be more suitable for machine learning
    # for this we use label encoding for categorical columns and standard scaling for numerical columns
    return_data = data.copy()
    label_encoders = {}
    categorical_columns = ['Crm Cd Desc', 'Vict Sex', 'Premis Desc']

    # we iterate through these columns and apply label encoding
    for column in categorical_columns:
        le = LabelEncoder()
        return_data.loc[:, column] = le.fit_transform(return_data[column])
        label_encoders[column] = le

    # we convert some columns to float64 to avoid issues with scaling
    return_data['TIME OCC'] = return_data['TIME OCC'].astype('float64')
    return_data['Vict Age'] = return_data['Vict Age'].astype('float64')

    # then we go on to scale the numerical columns
    scaler = StandardScaler()
    return_data.loc[:, 'TIME OCC'] = scaler.fit_transform(return_data[['TIME OCC']])
    return_data.loc[:, 'Vict Age'] = scaler.fit_transform(return_data[['Vict Age']])

    return return_data

def split_data(data, column):
    # y will be the column we're predicting, it's also called the target
    if column == 'Weapon Used Cd':
        y = data[column].apply(lambda x: 1 if x > 0 else 0)
    elif column == 'Crm Cd Desc':
        y = data[column].astype('category').cat.codes
    
    # X will be everything except the column we're predicting, it's also called the features
    X = data.drop(columns=[column])

    # now we split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Data preparation complete")
    print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")

    return X_train, X_test, y_train, y_test

def data_preparation():
    data = load(100000) # load the dataset
    data_filtered = filter(data) # filter the dataset
    handle_missing_values(data_filtered) # handle missing values
    data_featured = feature_engineering(data_filtered) # do feature engineering
    X_train_weapon, X_test_weapon, y_train_weapon, y_test_weapon = split_data(data_featured, 'Weapon Used Cd') # split the data into train and test sets for the weapon model
    X_train_crime, X_test_crime, y_train_crime, y_test_crime = split_data(data_featured, 'Crm Cd Desc') # split the data into train and test sets for the crime model
    return data_filtered, X_train_weapon, X_test_weapon, y_train_weapon, y_test_weapon, X_train_crime, X_test_crime, y_train_crime, y_test_crime

def model_weapon_training(X_train, X_test, y_train, y_test):
    # we train a decision tree classifier to predict if a weapon was used or not
    model_weapon = DecisionTreeClassifier(random_state=42)
    model_weapon.fit(X_train, y_train)

    # this is our prediction
    y_pred_weapon = model_weapon.predict(X_test)

    # let's evaluate the model's performance
    print("Weapon prediction - Classification report:")
    print(classification_report(y_test, y_pred_weapon))
    print("Weapon prediction - Accuracy:", accuracy_score(y_test, y_pred_weapon))
    
    return model_weapon

def model_weapon_visualize(model_weapon, X_test_weapon, y_test_weapon):
    # display the confusion matrix that shows how well the model predicts if a weapon was used or not
    ConfusionMatrixDisplay.from_estimator(model_weapon, X_test_weapon, y_test_weapon)
    plt.title("Weapon prediction confusion matrix")
    plt.show()

def model_crime_training(X_train, X_test, y_train, y_test):
    # filter the data to include only the crime types with at least threshold amount of support
    threshold = 20
    unique_classes = np.unique(y_test)
    class_support = {cls: sum(y_test == cls) for cls in unique_classes}
    valid_classes = [cls for cls, support in class_support.items() if support >= threshold]
    
    # create masks for valid classes
    train_mask = np.isin(y_train, valid_classes)
    test_mask = np.isin(y_test, valid_classes)
    
    # filter the data before training
    X_train_filtered = X_train[train_mask]
    y_train_filtered = y_train[train_mask]
    X_test_filtered = X_test[test_mask]
    y_test_filtered = y_test[test_mask]

    # we use a random forest classifier to predict the type of the crime
    model_crime_type = RandomForestClassifier(random_state=42)
    model_crime_type.fit(X_train_filtered, y_train_filtered)

    # this is our prediction
    y_pred_filtered = model_crime_type.predict(X_test_filtered)

    # let's evaluate the model's performance
    print("Crime type prediction - Classification report:")
    print(classification_report(y_test_filtered, y_pred_filtered, zero_division=1, labels=valid_classes))
    print("Crime type prediction - Accuracy:", accuracy_score(y_test_filtered, y_pred_filtered))

    return model_crime_type

def model_crime_visualize(model_crime_type, X_test_crime, y_test_crime):
    # number of different crime types, because too much will overload the confusion matrix
    N = 10
    
    # get the frequencies of these crime types 
    unique_types, type_counts = np.unique(y_test_crime, return_counts=True)

    # get the top N crime types
    top_n_types = unique_types[np.argsort(-type_counts)[:N]]
    
    # filter the data and leave only the rows with the crimes that we want
    mask = np.isin(y_test_crime, top_n_types)
    X_test_filtered = X_test_crime[mask]
    y_test_filtered = y_test_crime[mask]

    # predict the crime types
    y_pred_filtered = model_crime_type.predict(X_test_filtered)

    # make sure the predictions are only from top N crime types
    mask_pred = np.isin(y_pred_filtered, top_n_types)
    y_test_filtered = y_test_filtered[mask_pred]
    y_pred_filtered = y_pred_filtered[mask_pred]

    # map the test and prediction data to integers
    label_map = {label: idx for idx, label in enumerate(top_n_types)}

    y_test_mapped = np.array([label_map[label] for label in y_test_filtered])
    y_pred_mapped = np.array([label_map[label] for label in y_pred_filtered])
    
    # display the confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test_mapped,
        y_pred_mapped,
        display_labels=top_n_types,
        xticks_rotation=45,
        cmap='Blues'
    )
    disp.figure_.set_size_inches(8, 6)
    plt.title("Crime type prediction confusion matrix (top 10 classes)")
    plt.tight_layout()
    plt.show()

def remove_outliers(column):
    # remove outliers from the column using the IQR method
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return column[(column >= lower_bound) & (column <= upper_bound)]

def plot_visualizations(data):
    # visualize only numerical columns
    numeric_columns = data.select_dtypes(include='number').columns
    
    #remove the outliers
    for col in numeric_columns:
        data[col] = remove_outliers(data[col])

    numeric_data = data[numeric_columns]
    
    # histograms for numerical columns in the dataset
    numeric_data.hist(bins=30, edgecolor='black')
    plt.suptitle('Histograms of data')
    plt.show()

    # box plot for victim age
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data['Vict Age'])
    plt.title('Box plot of data')
    plt.show()

    # box plot for time of occurence
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data['TIME OCC'])
    plt.title('Box plot of data')
    plt.show()

    # histogram for crime frequency by hour
    plt.figure(figsize=(10, 6))
    sns.histplot(data['TIME OCC'], bins=24, kde=False, color='blue')
    plt.title('Crime frequency by hour')
    plt.xlabel('Hour of occurrence')
    plt.ylabel('Number of crimes')
    plt.show()

    # histogram for victim age distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Vict Age'], bins=20, kde=True, color='green')
    plt.title('Victim age distribution')
    plt.xlabel('Age')
    plt.ylabel('Number of victims')
    plt.show()

    # box plot for victim age distribution by top crime types
    plt.figure(figsize=(12, 8))
    top_crimes = data['Crm Cd Desc'].value_counts().head(5).index
    sns.boxplot(x='Crm Cd Desc', y='Vict Age', data=data[data['Crm Cd Desc'].isin(top_crimes)], palette='Set3', hue='Crm Cd Desc', legend=False)
    plt.title('Victim age distribution by top crime types')
    plt.xlabel('Crime type')
    plt.ylabel('Victim age')
    plt.xticks(rotation=45)
    plt.show()

    # histogram for crime frequency by area
    plt.figure(figsize=(10, 6))
    sns.histplot(data['AREA'], bins=15, kde=False, color='purple')
    plt.title('Crime frequency by area')
    plt.xlabel('Area')
    plt.ylabel('Number of crimes')
    plt.show()

def main():
    data_filtered, X_train_weapon, X_test_weapon, y_train_weapon, y_test_weapon, X_train_crime, X_test_crime, y_train_crime, y_test_crime = data_preparation()
    model_weapon = model_weapon_training(X_train_weapon, X_test_weapon, y_train_weapon, y_test_weapon)
    model_crime_type = model_crime_training(X_train_crime, X_test_crime, y_train_crime, y_test_crime)
    plot_visualizations(data_filtered)
    model_weapon_visualize(model_weapon, X_test_weapon, y_test_weapon)
    model_crime_visualize(model_crime_type, X_test_crime, y_test_crime)

if __name__ == "__main__":
    main()