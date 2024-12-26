Data of the project is not in the repo itself, since it is too big. You will need to add "data" folder in the project in order to use, that we will provide...

Documentation of Crime_Type_And_Weapon.py:

The task can be split into three different parts: clean, train, and plot.
  Clean: Cleaning uses the data_preparation function, which in turn uses other functions like filter, handle_missing_values, feature_engineering, split_data. The function also splits the data into train and test sets, this doesn't count as "cleaning" but it was put here for the simplicity of the overall code. As the names of the functions suggest, the data gets filtered, meaning it leaves only the necessary columns for the needed models, missing values get replaced with placeholder strings or numbers representing that the value is missing, then feature engineering is performed to transform the age and time of occurance columns. Finally the data is split two times, once for the weapon model and once for the crime type model.
  Train: Training is the section where the ML models are built and evaluated. There are two models, one involves weapons, the other involves crime types. Training of these models is done with the help of model_weapon_training and model_crime_training functions respectively. The first function uses a Decision tree classifier to predict if a weapon was used or not for a specific crime, then it evaluates the result by calculating precision, recall and accuracy of the model. Predicting if a weapon was used during a crime can help lead the law enforcment forces in the right direction. The second function uses a Random forest classifier to predict what time of crime was committed, but since there are a lot of crime types, the function first filters the data and leaves only the top N most common crimes. This is done in order to make the result more understandable and readable, namely the confusion matrix will be of size 10x10 instead of 100+ x 100+. The model works without this filtering too and it was done only for visualizing purposes. Finally, the model is evaluated the same way as the previous one, and it has less accuracy compared the weapon model, but it's understandable because the weapon model only has two options of the output, while the crime type model has a lot. Predicting the crime type might speed up the investigation or help the law enforcment forces better understand the situation overall.
  Plot: Plotting is the section where all the plots in this file happen. First, only the columns that have numeric values are left, then the outliers for the relevant plots are removed, then there are various plots about the original data, as well as the confusion matrices of the models built previously. The plots include: Histogram for columns representing the distribution of: Time of occurence, area, victim age, weapon used code; Box plot of victim age and time of occurence; Histogram of crime frequency by hour and victim age distribution; Box plot of victim age distribution by top 5 crime types; Histogram of crime frequency by area; Confusion matrices for both ML models.



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
