# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:21:27 2018

@author: chuny
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df=pd.read_csv('Iris.csv')

#dic={'Iris-setosa':0,
#     'Iris-versicolor':1,
#     'Iris-virginica':2
#     }
#df['Class']=df['Species'].map(dic)
#df=df.drop(['Id','Species','SepalWidthCm','PetalWidthCm'],axis=1)
df=df.drop(['Id','SepalWidthCm','PetalWidthCm'],axis=1)

X=df.values[:,0:2]
y=df.values[:,2]
plt.figure()
plt.scatter(X[:,0],X[:,1],marker='o',s=15)


x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
clf=KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)
accuracy=accuracy_score(y_test,y_predict)
print('accuracy:', accuracy )