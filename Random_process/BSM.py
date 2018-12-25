# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 14:36:34 2018

@author: chuny
"""


from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def call_option_pricer(spot,strike,maturity,r,vol):
    d1=(np.log(spot/strike)+(r+0.5*vol**2)*maturity)/vol/np.sqrt(maturity)
    d2=d1-vol*np.sqrt(maturity)
    price=spot*norm.cdf(d1)-strike*np.exp(-r*maturity)*norm.cdf(d2)
    return price



spot = 2.45
strike=2.5
maturity=0.25
r=0.05
vol=0.25

price=call_option_pricer(spot,strike,maturity,r,vol)

print('期权价格：', price)


portfolioSize = range(1, 100000, 500)
timeSpent = []
timeSpentNumpy = []
for size in portfolioSize:
    strikes = np.linspace(1,5.0, size)
    res = call_option_pricer(spot, strikes, maturity, r, vol)


plt.figure()
plt.plot(strikes,res)