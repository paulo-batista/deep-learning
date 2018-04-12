# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 13:29:04 2018

@author: a118905
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#loading the dataset, dependant and independant variable
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical (independant) data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features= [1])
X = onehotencoder.fit_transform(X).toarray()


