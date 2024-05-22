from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import warnings  # type: ignore
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings("ignore")
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def about(request):
    return render(request, 'about.html')


def result(request):
    data = pd.read_csv(r"static/DiabetesPrediction/dataset/diabetesdatasetmain.csv")

    X = data.drop(columns=["diabetes"], axis=1)
    Y = data["diabetes"]  # Target
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0, stratify=Y)

    # - To increase the size of the dataset by oversampling.
    smote = SMOTE(random_state=42)

    # X = data.drop("Outcome", axis=1)
    # Y = data['Outcome']
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # model = LogisticRegression()
    # model.fit(X_train, Y_train)

    #model = KNeighborsClassifier(n_neighbors=7)
    #model.fit(x_train, y_train)

    model = RandomForestClassifier(n_estimators=20, criterion='gini', max_depth=50)
    model.fit(x_train, y_train)

    val1 = int(request.GET['gender'])
    val2 = float(request.GET['n1'])
    val3 = int(request.GET['n2'])
    val4 = int(request.GET['n3'])
    val5 = int(request.GET['sh'])
    val6 = float(request.GET['n4'])
    val7 = float(request.GET['n5'])
    val8 = int(request.GET['n6'])

    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    result1 = ""
    if pred == [1]:
        result1 = "Positive"
    else:
        result1 = "Negative"
    return render(request, 'predict.html', {"result2": result1})
