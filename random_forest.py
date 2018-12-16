from math import log
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import numpy as np

def transform_lable(data):
    data = data.apply(lambda x: x.replace(['?'], np.nan))
    data = data.apply(lambda x: x.fillna(x.value_counts().index[0]))
    dict = defaultdict(LabelEncoder)
    data = data.apply(lambda x: dict[x.name].fit_transform(x))
    return data

def preprocess_data(path):
    data = pd.read_csv(path, header=None)
    data = transform_lable(data)
    x_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]
    return x_data, y_data

def applay_model(x_data, y_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=24)

    my_n_estimators = 100
    my_max_depth = log(len(x_train.columns)) + 1
    my_min_samples_split = int(2*x_train.shape[0] / (len(x_train.columns)))
    my_max_features = None
    my_min_samples_leaf = len(y_data.unique()) * 10

    rf = RandomForestClassifier(n_estimators=my_n_estimators, max_depth=my_max_depth, min_samples_split=my_min_samples_split,
                                max_features=my_max_features, min_samples_leaf=my_min_samples_leaf)

    rf.fit(x_train, y_train)
    gnm_pred = rf.predict(x_test)

    gnm_score = sum(i == 0 for i in (gnm_pred-y_test))/len(gnm_pred)
    print(gnm_score)
    return gnm_score

x_data, y_data = preprocess_data(r'.\data base\chess.csv') #C:\Users\User\Documents\information enginer\sem 7\machine lerning\assignment2

scores_sum = 0
for i in range(10):
    score = applay_model(x_data, y_data)
    scores_sum = scores_sum + score

print("score: ")
print(scores_sum/10)
