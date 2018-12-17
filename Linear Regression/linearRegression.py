# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 09:29:06 2018

@author: chuny
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#from sklearn.model_selection import train_test_split


df_train=pd.read_csv('train.csv')
#,skiprows=1,names=['x','y']
X_train=np.array(df_train.x).reshape(-1,1)
y_train=np.array(df_train.y)

df_test=pd.read_csv('test.csv',skiprows=1,names=['x','y'])
X_test=np.array(df_test.x).reshape(-1,1)
y_test=np.array(df_test.y)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
ax1.scatter(X_train,y_train,color='k',linewidth=2,s=55,edgecolors='y',label='Train sample')
ax2.scatter(X_test,y_test,color='r',linewidth=2,s=55,edgecolors='b',marker='*',label='Test sample')


clf=LinearRegression()
clf.fit(X_train,y_train)
y_train_predict=clf.predict(X_train)
ax1.plot(X_train,y_train_predict,label='Fitted line',linewidth=3)

y_test_predict=clf.predict(X_test)
score=r2_score(y_test,y_test_predict)
print('R2 score:', score)
ax2.plot(X_test,y_test_predict,label='Fitted line',linewidth=3)

ax1.legend()
ax2.legend()
ax1.set(xlabel='X',ylabel='y')
ax2.set(xlabel='X',ylabel='y')

plt.figure()
plt.scatter(X_test,y_test_predict-y_test)
plt.hlines(y=0,xmin=0,xmax=100)
plt.ylabel('Residual')
plt.xlim(0,100)
plt.ylim(-10,10)



##linear regression by chuny
#n=700
#a_0=np.zeros((n,1))
#a_1=np.zeros((n,1))
#alpha=0.00001
#mean_sq_error=1
#count=1
#X_train.reshape(1,-1)
#while(count<10):
#    y=a_0+a_1*X_train
#    error=y-y_train
#    mean_sq_error=np.square(error).sum()/n
#    a_0=a_0-alpha*2*np.sum(error)/n
#    a_1=a_1-alpha*2*np.sum(error*X_train)/n
#    count+=1
#
