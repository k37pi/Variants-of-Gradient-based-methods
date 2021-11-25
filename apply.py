#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 00:38:45 2021

@author: kshitij
"""
import functions as fn
import dms 

from inspect import getmembers, isfunction

avail_fns = []

for i in dir(fn):
  if type(getattr(fn,i)) == list:
    avail_fns.append(i)
    print(i,"  ",type(getattr(fn,i)))


name = "freudroth"

f_all = fn.goldsteinprice
f = f_all[0]
xmin, xmax, xstep = f_all[2][0][0], f_all[2][0][1], .2
ymin, ymax, ystep = f_all[2][1][0], f_all[2][1][1], .2
search = asarray([[xmin, xmax], [ymin, ymax]])

x0 = np.array(search[:, 0] + rand(len(search)) * (search[:, 1] - search[:, 0]))#np.array([3.0,4.0])
#x0 = np.array([29.0,59.0])

eps = 10**-5

methods = [("Gradient",grad_descent(f,x0,eps,"BT",a=0.5,b=0.5,s=1))]
methods.append(("Stochastic Gradient", sgd(f,x0,eps,search,"Constant",batch = 50,t=0.9,a=0.5,b=0.5,s=1)))
methods.append(("Nesterov",nesterov(f,x0,eps,tk = 0.05, momentum = 0.9)))
methods.append(("Adagrad", adagrad(f, x0, eps, tk=0.01)))
methods.append(("Adam",adam(f, x0, eps, t = 0.02, beta1 = 0.8, beta2 = 0.99)))

[print(methods[i][0]) for i in range(len(methods))] # Method names.
[print(methods[i][1][0]) for i in range(len(methods))] # number of iterations per method.

# Iterate paths 
x0_ = x0.reshape(-1,1)
path = [np.array(methods[i][1][1]).T for i in range(len(methods))]
len(path)
[path[i].shape for i in range(len(path))]

# Grid
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = f(x,y)
minima = np.array(f_all[1])
minima_ =   minima.T

dz_dx = elementwise_grad(f, argnum=0)(x, y)
dz_dy = elementwise_grad(f, argnum=1)(x, y)

# colours
col = ['k','r','g','b','y']

# 2d lev curves with directions
fig, ax = plt.subplots(figsize=(10, 6))

ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
#ax.contour(x, y, z, levels=50, cmap='jet')
for i in range(len(path)):
  ax.quiver(path[i][0,:-1], path[i][1,:-1], path[i][0,1:]-path[i][0,:-1], path[i][1,1:]-path[i][1,:-1], scale_units='xy', angles='xy', scale=1,zorder = 1, color=col[i],label = str(methods[i][0])+" "+ str(methods[i][1][0])) 
  
ax.plot(*minima_, 'm*', markersize=18,label = "Minima")
ax.plot(*x0_, 'co', markersize=9.7,label = "x0")

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.title('Goldstein Price')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

plt.legend(loc='upper left')


# 3d plot with directions
fig = plt.figure(figsize=(8, 5))
ax = plt.axes(projection='3d', elev=50, azim=-50)

ax.plot_surface(x, y, z,cmap=plt.cm.jet)# norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)
for i in range(len(path)):
  ax.quiver(path[i][0,:-1], path[i][1,:-1], f(*path[i][::,:-1]), 
          path[i][0,1:]-path[i][0,:-1], path[i][1,1:]-path[i][1,:-1], f(*(path[i][::,1:]-path[i][::,:-1])), 
          color=col[i],label = str(methods[i][0])+" "+ str(methods[i][1][0]))

ax.plot(*minima_, f(*minima_), 'm*', markersize=10)
ax.plot(*x0_, f(*x0_), 'co', markersize=9.7,label = "x0")

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
plt.legend(loc='upper left')
