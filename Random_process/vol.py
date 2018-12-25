# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:58:46 2018

@author: chuny
"""

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def call_option_pricer(spot,strike,maturity,r,vol):
    d1=(np.log(spot/strike)+(r+0.5*vol**2)*maturity)/vol/np.sqrt(maturity)
    d2=d1-vol*np.sqrt(maturity)
    price=spot*norm.cdf(d1)-strike*np.exp(-r*maturity)*norm.cdf(d2)
    return price


#
#spot = 2.45
#strike=2.5
#maturity=0.25
#r=0.05
#vol=0.25


class cost_function:
    def __init__(self, target):
        self.targetValue = target
    def __call__(self, x):
        return call_option_pricer(spot, strike, maturity, r, x)- self.targetValue
# 假设我们使用vol初值作为目标
target = call_option_pricer(spot, strike, maturity, r, vol)
cost_sampel = cost_function(target)
# 使用Brent算法求解
impliedVol = brentq(cost_sampel, 0.01, 0.5)
print (u'真实波动率： %.2f' % (vol*100,) + '%')
print (u'隐含波动率： %.2f' % (impliedVol*100,) + '%')
