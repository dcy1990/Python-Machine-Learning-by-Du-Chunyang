# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 19:47:03 2018

@author: chuny
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('Iris.csv')
df=df.drop(['Id'],axis=1)

X=df.values[:,0:4][1,1]
y=df.values[:,4]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4)
#
clf=SVC(kernel='linear')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))