Data of the project is not in the repo itself, since it is too big. You will need to add "data" folder in the project in order to use, that we will provide...

# Crime_Type_And_Weapon

## Overview

The `Crime_Type_And_Weapon.py` file tackles the problem of predicting whether a weapon was used in a crime and identifying the type of crime committed, based on various factors. This project is divided into three key sections: **Clean**, **Train**, and **Plot**. Each part of the process ensures that the data is properly handled, the models are accurately trained, and the results are visually interpreted.

## Table of Contents
1. [Clean](#clean)
2. [Train](#train)
3. [Plot](#plot)
4. [Evaluation](#evaluation)
5. [Real-World Application](#real-world-application)

---

## Clean

The cleaning process prepares the data for model training, splitting it into train and test sets for two distinct tasks: weapon prediction and crime type prediction. The data preparation steps include:

1. **Data Filtering**: 
   - Removes unnecessary columns to keep only the relevant features needed for both models.

2. **Handling Missing Values**: 
   - Missing values are replaced with placeholder strings or numbers that indicate missing data.

3. **Feature Engineering**:
   - The `age` and `time of occurrence` columns are transformed for better model performance.

4. **Data Splitting**: 
   - The data is split into training and testing sets for both the weapon prediction model and the crime type prediction model. Although splitting is a form of data partitioning, it is categorized as part of the cleaning process for simplicity.

---

## Train

### Weapon Prediction Model:
- **Algorithm**: Decision Tree Classifier
- **Goal**: Predict whether a weapon was used in a crime.
- **Evaluation Metrics**: Precision, Recall, and Accuracy.

### Crime Type Prediction Model:
- **Algorithm**: Random Forest Classifier
- **Goal**: Predict the type of crime (filtered to the top N most common crimes for clarity).
- **Evaluation Metrics**: Precision, Recall, and Accuracy.
- **Note**: The crime type model has more classes to predict, which may result in slightly lower accuracy compared to the weapon model, but it provides critical insights for law enforcement.

---

## Plot

The plotting section visualizes both the raw data and model results, enabling a comprehensive understanding of the data and model performance.

### Plots Included:
1. **Histograms**:
   - Distribution of `Time of Occurrence`
   - Distribution of `Area`
   - Distribution of `Victim Age`
   - Frequency of `Weapon Used Code`
   - Crime frequency by `Hour`

2. **Box Plots**:
   - `Victim Age`
   - `Time of Occurrence`
   - Victim age distribution by the top 5 crime types

3. **Confusion Matrices**:
   - One for the **Weapon Prediction Model**
   - One for the **Crime Type Prediction Model**

### Data Preprocessing for Plots:
- Only numeric columns are retained for plotting.
- Outliers in the relevant plots are removed to ensure more accurate visualizations.

---

## Evaluation

The evaluation of the models involves calculating key performance metrics such as precision, recall, and accuracy. The results of the confusion matrices provide further insights into model performance. While the weapon model has a higher accuracy due to fewer output classes (two options), the crime type model can still offer significant insights despite a higher number of output classes.

---

## Real-World Application

### Weapon Prediction:
- **Impact**: Helps law enforcement identify whether a weapon was involved in a crime, guiding the investigation and improving safety protocols.

### Crime Type Prediction:
- **Impact**: Quickly categorizes crime types, speeding up law enforcement responses and improving overall situational understanding.

---

## Conclusion

This project demonstrates how machine learning can be applied to crime data to assist law enforcement in predicting critical information, such as the involvement of a weapon and the type of crime. By leveraging decision trees and random forests, the models provide actionable insights that can enhance investigation efficiency.

Feel free to explore the code and try it on your own dataset to gain more insights into crime prediction!

---

# Crime Analysis and Prediction System
A system for analyzing crime data and predicting future crime patterns using machine learning techniques. This project combines Random Forest classification for crime type prediction and ARIMA modeling for time-series forecasting.

# Features
1. Data Preprocessing
    
    Handles datetime conversion and formatting
    Filters crimes based on occurrence frequency
    Handles missing values intelligently
    Preprocesses data for machine learning compatibility


2. Crime Classification
   
    Implements Random Forest classification
    Predicts crime types based on multiple features
    Includes victim demographics, location, and circumstantial data
    Provides accuracy analysis for different crime types


3. Time Series Forecasting
   
    Uses ARIMA modeling for crime prediction
    Forecasts crime occurrences for the next 15 days
    Visualizes historical and predicted crime trends


4. Visualization & Reporting
   
    Generates comparative analysis of actual vs. predicted crimes
    Creates visual representations of prediction accuracy
    Produces formatted tables for easy data interpretation
    Includes time series visualization of crime forecasts


# Crime Hotspot Analysis System
A system for analyzing crime patterns and identifying potential hotspots using machine learning techniques. The project uses K-means clustering for hotspot detection and Random Forest classification for pattern prediction, with interactive map visualizations using Folium.

# Features

1. Data Preprocessing

    Datetime handling and formatting
    Cyclical time feature encoding
    Geographic coordinate validation
    Missing value handling


2. Current Crime Visualization

    Interactive heat map of current crime distribution
    Color-coded crime types
    Tooltip information for each crime point
    Automatic map centering based on data


3. Hotspot Detection

    K-means clustering to identify crime hotspots
    Time-aware clustering using both location and temporal features
    Calculation of crime proportions per hotspot
    Cluster visualization on interactive maps


4. Pattern Prediction
  
    Random Forest classification for pattern prediction
    Temporal feature-based learning
    Accuracy assessment and comparison
    Cluster prediction capabilities


# Crime Data Visualization
A functionality for visualization, analyzing crime data through various statistical and graphical representations. It provides multiple visualization functions to help understand crime patterns, demographics, and trends.

# Features
# Temporal Analysis

Crime Trend Over Time

1. Daily crime frequency plotting
2. 7-day moving average
3. Interactive time series visualization


Time-Based Patterns

1. Hour-of-day vs day-of-week heatmap
2. Yearly crime frequency analysis
3. Monthly crime distribution



# Demographic Analysis

Victim Demographics

1. Age distribution with density plots
2. Gender distribution pie charts
3. Descent (ethnicity) distribution analysis



# Crime Analysis

Crime Types

1. Top 10 most common crimes
2. Multi-class crime analysis
3. Crime frequency visualization


Crime Characteristics

1. Weapon usage analysis
2. Premises type analysis
3. Case status and resolution rates
