# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:05:53 2018

@author: chuny
"""

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

delta=0.25
dt=0.01
x=50
x2=50
n=10000
seed=7
xs=[]
ts=[]
xs2=[]
ts2=[]
mu=0.001
for k in range(n):
    tmp=norm.rvs(scale=delta**2*dt,)
    x=x+tmp*x
    xs.append(x)
    ts.append(k*dt)
#    x2=x2+tmp*x2+mu*x2*dt
#    xs2.append(x2)
#    ts2.append(k*dt)
#    

plt.figure()
plt.plot(ts,xs,'r')
#plt.plot(ts2,xs2,'k')
'''
def Brownian_move(x0,n,dt,delta,out=None):
    x0=np.asarray(x0)
    r=norm.rvs(size=x0.shape+(n,),scale=delta*np.sqrt(dt))
    if out is None:
        out = np.empty(r.shape)
    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)
    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out


# The Wiener process parameter.
delta = 2
# Total time.
T = 10000.0
# Number of steps.
N = 500000
# Time step size
dt = T/N
# Number of realizations to generate.
m = 20
# Create an empty array to store the realizations.
x = np.empty((m,N+1))
# Initial values of x.
x[:, 0] = 50

Brownian_move(x[:,0], N, dt, delta, out=x[:,1:])

t = np.linspace(0.0, N*dt, N+1)
plt.figure()
for k in range(m):
    plt.plot(t, x[k])
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.grid(True)
plt.show()


plt.figure()
plt.plot(x[0],x[1])
'''