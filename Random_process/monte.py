# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:03:12 2018

@author: chuny
"""

import scipy
import numpy as np
from matplotlib import pylab

spot = 2.45
strike=2.5
maturity=0.25
r=0.05
vol=0.25


def call_option_pricer_monte_carlo(spot, strike, maturity, r, vol, numOfPath = 50000):
    randomSeries = scipy.random.randn(numOfPath)
    s_t = spot * np.exp((r - 0.5 * vol * vol) * maturity + randomSeries * vol * np.sqrt(maturity))
    sumValue = np.maximum(s_t - strike, 0.0).sum()
    price = np.exp(-r*maturity) * sumValue / numOfPath
    return price


price=call_option_pricer_monte_carlo(spot, strike, maturity, r, vol)

print('期权价格（蒙特卡洛）：', price)

pathScenario = range(1000, 50000, 1000)
numberOfTrials = 100
confidenceIntervalUpper = []
confidenceIntervalLower = []
means = []
for scenario in pathScenario: 
    res = np.zeros(numberOfTrials)
    for i in range(numberOfTrials):
        res[i] = call_option_pricer_monte_carlo(spot, strike, maturity, r, vol, numOfPath = scenario)
    means.append(res.mean())
    confidenceIntervalUpper.append(res.mean() + 1.96*res.std())
    confidenceIntervalLower.append(res.mean() - 1.96*res.std())

pylab.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签  
pylab.figure(figsize = (12,8))
tabel = np.array([means,confidenceIntervalUpper,confidenceIntervalLower]).T
pylab.plot(pathScenario, tabel)
pylab.title(u'期权计算蒙特卡洛模拟',  fontsize = 18)
pylab.legend([u'均值', u'95%置信区间上界', u'95%置信区间下界'])
pylab.ylabel(u'价格',  fontsize = 15)
pylab.xlabel(u'模拟次数',  fontsize = 15)
pylab.grid(True)