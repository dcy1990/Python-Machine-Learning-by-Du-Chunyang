# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:57:17 2018

@author: chuny
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns

sns.set(style="ticks")


df=pd.read_csv('Iris.csv')
df=df.drop(['Id'],axis=1)
#sns.pairplot(df, hue="Species")
df=df[0:99]
#sns.pairplot(df, hue="Species")
#pd.plotting.scatter_matrix(df)

X=np.array(df[['PetalWidthCm','PetalLengthCm']])
y=(np.array(df['Species']=='Iris-setosa')*1)

plt.figure()
plt.scatter(X[:,0],X[:,1],c=y,marker='o',s=15)

clf=LogisticRegression()
clf.fit(X,y)
prediction=clf.predict(X)
print('coef:',clf.coef_)
print('accuracy:',accuracy_score(y,prediction))


prediction2=1/(1+np.exp(-np.dot(clf.coef_,X.T)-clf.intercept_))


plt,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
ax1.scatter(X[:,0],X[:,1],c=prediction2[0,:],marker='o',s=55,label='calculated',cmap='summer')
ax2.scatter(X[:,0],X[:,1],c=prediction,marker='s',s=55,label='function',cmap='summer')
ax1.legend()
ax2.legend()
