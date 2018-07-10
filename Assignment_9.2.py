# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 16:18:13 2018

@author: vamshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target

#Divide train and test the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test=train_test_split(features, targets, test_size=0.29, random_state=1)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
y_train = np.array(y_train).reshape(-1,1)
y_train = sc1.fit_transform(y_train).astype(int)
y_test=sc1.transform(y_test)

#Fitting rando forrest Classsifier 

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=5, criterion='mse', random_state=0)
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
cm=confusion_matrix(y_test,y_pred)

accuracy=accuracy_score(y_test,y_pred)
clas=classification_report(y_test,y_pred)


