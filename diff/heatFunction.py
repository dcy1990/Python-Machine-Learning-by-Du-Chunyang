# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 12:27:08 2018

@author: chuny
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号



N=25
M=2500
T=1.0
X=1.0
def initialCondition(x):
    return 4.0*(1.0 - x) * x

xArray=np.linspace(0,X,N+1)
yArray=list(map(initialCondition,xArray))

plt.figure(figsize = (12,6))
plt.plot(xArray, yArray)

startValues=yArray
U=np.zeros((N+1,M+1))
U[:,0]=startValues
dx=X/N
dt=T/M
kappa=1
rho=kappa*dt/dx/dx

for k in range(0,M):
    for j in range(1,N):
        U[j][k+1]=rho*U[j-1][k]+(1-2*rho)*U[j][k]+rho*U[j+1][k]
    U[0][k+1]=0
    U[N][k+1]=0
plt.figure(figsize = (12,6))
plt.plot(xArray, U[:,0])
plt.plot(xArray, U[:, int(0.10/ dt)])
plt.plot(xArray, U[:, int(0.20/ dt)])
plt.plot(xArray, U[:, int(0.50/ dt)])
plt.xlabel('$x$', fontsize = 15)
plt.ylabel(r'$U(\dot, \tau)$', fontsize = 15)
plt.title(u'一维热传导方程')
plt.legend([r'$\tau = 0.$', r'$\tau = 0.10$', r'$\tau = 0.20$', r'$\tau = 0.50$'], fontsize = 15)


tArray = np.linspace(0, 0.2, int(0.2 / dt) + 1)
xGrids, tGrids = np.meshgrid(xArray, tArray)
fig= plt.figure(figsize = (10,6))
ax = fig.add_subplot(1, 1, 1, projection = '3d')
surface = ax.plot_surface(xGrids, tGrids, U[:,:int(0.2 / dt) + 1].T, cmap=cm.coolwarm)
ax.set_xlabel("$x$", fontdict={"size":18})
ax.set_ylabel(r"$\tau$", fontdict={"size":18})
ax.set_zlabel(r"$U$", fontdict={"size":18})
ax.set_title(u"热传导方程 $u_\\tau = u_{xx}$")
fig.colorbar(surface,shrink=0.75)


class HeatEquation:
    def __init__(self, kappa, X, T,initialConstion = lambda x:4.0*x*(1.0-x), 
             boundaryConditionL = lambda x: 0, boundaryCondtionR = lambda x:0):
        self.kappa = kappa
        self.ic = initialConstion
        self.bcl = boundaryConditionL
        self.bcr = boundaryCondtionR
        self.X = X
        self.T = T

class ExplicitEulerScheme:
    def __init__(self, M, N, equation):
        self.eq = equation
        self.dt = self.eq.T / M
        self.dx = self.eq.X / N
        self.U = np.zeros((N+1, M+1))
        self.xArray = np.linspace(0,self.eq.X,N+1)
        self.U[:,0] = list(map(self.eq.ic, self.xArray))
        self.rho = self.eq.kappa * self.dt / self.dx / self.dx
        self.M = M
        self.N = N
    def roll_back(self):
        for k in range(0, self.M):
            for j in range(1, self.N):
                self.U[j][k+1] = self.rho * self.U[j-1][k] + (1.
                      - 2*self.rho) * self.U[j][k] + self.rho * self.U[j+1][k]
                self.U[0][k+1] = self.eq.bcl(self.xArray[0])
                self.U[N][k+1] = self.eq.bcr(self.xArray[-1])
    def mesh_grids(self):
        tArray = np.linspace(0, self.eq.T, M+1)
        tGrids, xGrids = np.meshgrid(tArray, self.xArray)
        return tGrids, xGrids
    
ht = HeatEquation(1.,1.,1.)
scheme = ExplicitEulerScheme(2500,25, ht)
scheme.roll_back()

tGrids, xGrids = scheme.mesh_grids()
fig= plt.figure(figsize = (10,6))
ax = fig.add_subplot(1, 1, 1, projection = '3d')
cutoff = int(0.2 / scheme.dt) + 1
surface = ax.plot_surface(xGrids[:,:cutoff], tGrids[:,:cutoff],
scheme.U[:,:cutoff], cmap=cm.coolwarm)
ax.set_xlabel("$x$", fontdict={"size":18})
ax.set_ylabel(r"$\tau$", fontdict={"size":18})
ax.set_zlabel(r"$U$", fontdict={"size":18})
ax.set_title(u"热传导方程 $u_\\tau = u_{xx}$")
fig.colorbar(surface,shrink=0.75)