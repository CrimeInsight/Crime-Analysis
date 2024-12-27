from analysis import Analysis


class CrimeRecidivism(Analysis):
    def __init__(self, path):
        super().__init__(path)

    def clean(self):
        try:
            self.df = self.pd.read_csv(self.path)
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return  

        try:
            self.df.fillna(self.df.median(numeric_only=True), inplace=True)
            self.df.fillna('Unknown', inplace=True)
        except Exception as e:
            print(f"Error handling missing values: {e}")
            return  

        try:
            for col in ['Age at Release', 'Release Year']:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                self.df[col] = self.np.clip(self.df[col], lower_bound, upper_bound)
        except Exception as e:
            print(f"Error handling outliers for columns {col}: {e}")
            return


    def train(self):
        label_encoder = self.LabelEncoder()
        for col in ['County of Indictment', 'Gender', 'Crime Type']:
            self.df[col] = label_encoder.fit_transform(self.df[col])

        scaler = self.StandardScaler()
        self.df[['Age at Release', 'Release Year']] = scaler.fit_transform(
            self.df[['Age at Release', 'Release Year']]
        )

        X = self.df[['Release Year', 'County of Indictment', 'Gender', 'Age at Release', 'Crime Type']]
        y = self.df['Return Status']

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Logistic Regression
        self.logistic_model = self.LogisticRegression()
        self.logistic_model.fit(self.X_train, self.y_train)
        self.y_pred_logistic = self.logistic_model.predict(self.X_test)
        self.logistic_accuracy = self.accuracy_score(self.y_test, self.y_pred_logistic)

        # Random Forest
        self.rf_model = self.RandomForestClassifier(random_state=42)
        self.rf_model.fit(self.X_train, self.y_train)
        self.y_pred_rf = self.rf_model.predict(self.X_test)
        self.rf_accuracy = self.accuracy_score(self.y_test, self.y_pred_rf)

    def plot(self):
        # Exploratory Data Analysis

        # Crime Type Distribution
        crime_type_counts = self.df['Crime Type'].value_counts()
        self.plt.figure(figsize=(8, 8))
        self.plt.pie(
            crime_type_counts,
            labels=crime_type_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=self.plt.cm.Paired.colors
        )
        self.plt.title('Crime Type Distribution')
        self.plt.show()

        # Age Distribution
        self.plt.figure(figsize=(8, 6))
        self.sns.histplot(self.df['Age at Release'], kde=True, color='blue', bins=20)
        self.plt.title('Age Distribution')
        self.plt.xlabel('Age at Release')
        self.plt.ylabel('Frequency')
        self.plt.show()

        # Gender Distribution
        gender_counts = self.df['Gender'].value_counts()
        self.plt.figure(figsize=(6, 6))
        self.sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='viridis')
        self.plt.title('Gender Distribution')
        self.plt.ylabel('Count')
        self.plt.xlabel('Gender')
        self.plt.show()

        # Correlation Heatmap
        df_corr = self.df.copy()
        label_encoder = self.LabelEncoder()
        df_corr['Gender'] = label_encoder.fit_transform(df_corr['Gender'])
        df_corr['County of Indictment'] = label_encoder.fit_transform(df_corr['County of Indictment'])
        df_corr['Crime Type'] = label_encoder.fit_transform(df_corr['Crime Type'])
        df_corr['Return Status'] = label_encoder.fit_transform(df_corr['Return Status'])

        self.plt.figure(figsize=(12, 10))
        correlation_matrix = df_corr[[
            'Release Year', 'County of Indictment', 'Gender', 'Age at Release', 'Return Status', 'Crime Type'
        ]].corr()
        self.sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        self.plt.title('Correlation Heatmap (All Features)')
        self.plt.tight_layout()
        self.plt.show()

        # Feature Importance (Random Forest)
        feature_importances = self.pd.DataFrame(
            self.rf_model.feature_importances_,
            index=self.X_train.columns,
            columns=['Importance']
        ).sort_values(by='Importance', ascending=False)

        self.plt.figure(figsize=(8, 6))
        self.sns.barplot(x=feature_importances['Importance'], y=feature_importances.index, palette='cool')
        self.plt.title('Feature Importance (Random Forest)')
        self.plt.show()

        # Model Comparison
        model_accuracies = self.pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest'],
            'Accuracy': [self.logistic_accuracy, self.rf_accuracy]
        })

        self.plt.figure(figsize=(8, 6))
        self.sns.barplot(x='Model', y='Accuracy', data=model_accuracies, palette='viridis')
        self.plt.title('Model Accuracy Comparison')
        self.plt.xlabel('Model')
        self.plt.ylabel('Accuracy')
        self.plt.ylim(0, 1)
        self.plt.show()
