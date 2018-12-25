# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 21:07:30 2018

@author: chuny
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df=pd.read_csv('sonar-all-data.csv',header=None)
df_values = df.values
X = df_values[:,0:60].astype(float)
Y = df_values[:,60]
df2=df[[1,2,60]]
sns.pairplot(df2,hue=60)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
plt.show()