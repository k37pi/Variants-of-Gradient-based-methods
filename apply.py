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


methods = []
for i in dir(dms):
  if type(getattr(dms,i)) == 'function':
  #avail_fns.append(i)
    print(i,"  ",type(getattr(dms,i)))


f_all = fn.beales
f = f_all[0]
xmin, xmax, xstep = f_all[2][0][0], f_all[2][0][1], .2
ymin, ymax, ystep = f_all[2][1][0], f_all[2][1][1], .2
bounds = asarray([[xmin, xmax], [ymin, ymax]])

x0 = np.array(bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0]))#np.array([3.0,4.0])
eps = 10**-5

k, xk_gd = grad_descent(f,eps,"BT",x0,a=0.5,b=0.5,s=1)
xk_nest = nesterov(f,bounds,K=100,tk=0.02,momentum=0.78)
xk = adagrad(f,bounds,K=1000,tk=0.5)

x0_ = x0.reshape(-1,1)
x0_ = xk[0].reshape(-1,1)

path = np.array(xk).T
path.shape

#xmin, xmax, xstep = -60, 10, .2
#ymin, ymax, ystep = -6, 8, .2
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
#z = fn.ackley(x,y)
z = f(x,y)
minima = np.array(f_all[1])
minima_ =   minima.T

dz_dx = elementwise_grad(f, argnum=0)(x, y)
dz_dy = elementwise_grad(f, argnum=1)(x, y)


# 2d lev curves with directions
fig, ax = plt.subplots(figsize=(10, 6))

ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
ax.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], scale_units='xy', angles='xy', scale=1, color='k')
#ax.plot(*xk, 'c+', markersize=18)
ax.plot(*minima_, 'r*', markersize=18)
ax.plot(*x0_, 'bo', markersize=9.7)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

ax.set_xlim((xmin, xmax))
#ax.set_ylim((-6, 8))
ax.set_ylim((ymin, ymax))

plt.legend(loc='upper left')


# 3d plot with directions
fig = plt.figure(figsize=(8, 5))
ax = plt.axes(projection='3d', elev=50, azim=-50)

ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)
ax.quiver(path[0,:-1], path[1,:-1], f(*path[::,:-1]), 
          path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], f(*(path[::,1:]-path[::,:-1])), 
          color='k')
ax.plot(*minima_, f(*minima_), 'r*', markersize=10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

# 2d anim -------
fig, ax = plt.subplots(figsize=(10, 6))

ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
ax.plot(*minima_, 'r*', markersize=18)

line, = ax.plot([], [], 'b', label='Grad. Desc.', lw=2)
point, = ax.plot([], [], 'bo')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

ax.legend(loc='upper left')

def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point
  
def animate(i):
    line.set_data(*path[::,:i])
    point.set_data(*path[::,i-1:i])
    return line, point 
  
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=path.shape[1], interval=60, 
                               repeat_delay=5, blit=True)  

HTML(anim.to_html5_video())
