# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:26:26 2018

@author: chuny
"""

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url,names=names)
print(dataset.describe())

print(dataset.groupby('class').size())

dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
dataset.hist()

pd.plotting.scatter_matrix(dataset)

array=dataset.values
X=array[:,0:4]
y=array[:,4]
valadation_size=0.2
seed=7
scoring='accuracy'
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=valadation_size,random_state=seed)


models=[]
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results=[]
names=[]

for name, model in models:
    kfold=model_selection.KFold(n_splits=5,random_state=seed)
    cv_results=model_selection.cross_val_score(model,X_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print('%s:%f (%f)' %(name,cv_results.mean(),cv_results.std()))
    
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
predictions=knn.predict(X_test)
print(accuracy_score(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
