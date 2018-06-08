# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 09:10:14 2018

@author: vamshi
"""


import numpy as np
import pandas as pd
import matplotlib as plt
#pd.read_csv()
url='https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
dataset =pd.read_csv(url, usecols=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare','Survived'] )

X=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values

#take care of missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(strategy='mean', axis=0)
impuetr=imputer.fit(X[:,[2]])
X[:,[2]]=imputer.transform(X[:,[2]])

# Encoding the categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X[:,1]=labelencoder_X.fit_transform(X[:,1])

#Feature Scalling ploting the data with in the same range


#Divide train and test the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test=train_test_split(X, y, test_size=0.35, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Create the mpodel

from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)


from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
cm=confusion_matrix(y_test,y_pred)

accuracy=accuracy_score(y_test,y_pred)
clas=classification_report(y_test,y_pred)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Create the mpodel

from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)


from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
cm=confusion_matrix(y_test,y_pred)

accuracy=accuracy_score(y_test,y_pred)
clas=classification_report(y_test,y_pred)






