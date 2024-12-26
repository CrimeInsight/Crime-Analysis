import pandas as pd
import folium
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Analysis:
    def __init__(self, path="data\Crime_Data_from_2020_to_Present.csv"):
        # Shared dependencies
        self.pd = pd
        self.folium = folium
        self.train_test_split = train_test_split
        self.LabelEncoder = LabelEncoder
        self.StandardScaler = StandardScaler
        self.DecisionTreeClassifier = DecisionTreeClassifier
        self.RandomForestClassifier = RandomForestClassifier
        self.ConfusionMatrixDisplay = ConfusionMatrixDisplay
        self.classification_report = classification_report
        self.accuracy_score = accuracy_score
        self.np = np
        self.plt = plt
        self.sns = sns
        self.LogisticRegression = LogisticRegression

        self.path = path

    def execute(self):
        self.clean()
        self.train()
        self.plot()

    def clean(self):
        raise NotImplementedError("Subclasses must implement this method")

    def train(self):
        raise NotImplementedError("Subclasses must implement this method")

    def plot(self):
        raise NotImplementedError("Subclasses must implement this method")
