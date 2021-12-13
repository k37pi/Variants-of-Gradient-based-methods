#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 00:38:45 2021

@author: kshitij
"""
import functions as fn
import dms 

from inspect import getmembers, isfunction
import time

import csv
import pandas as pd
from pathlib import Path
import os 

import seaborn as sns


avail_fns = []

for i in dir(fn):
  if type(getattr(fn,i)) == list:
    avail_fns.append(i)
    print(i,"  ",type(getattr(fn,i)))



name = 'styblinskitang'

#f_all = fn.goldsteinprice
f_all = getattr(fn,name)
f = f_all[0]
xmin, xmax, xstep = f_all[2][0][0], f_all[2][0][1], .2
ymin, ymax, ystep = f_all[2][1][0], f_all[2][1][1], .2
search = asarray([[xmin, xmax], [ymin, ymax]])

'''
x0 = np.array(search[:, 0] + rand(len(search)) * (search[:, 1] - search[:, 0]))#np.array([3.0,4.0])
#x0 = np.array([29.0,59.0])

methods = [("Gradient",grad_descent(f,x0,eps,"BT",a=0.5,b=0.5,s=1))]
methods.append(("Stochastic Gradient", sgd(f,x0,eps,search,"Constant",batch = 50,t=0.9,a=0.5,b=0.5,s=1)))
methods.append(("Nesterov",nesterov(f,x0,eps,tk = 0.05, momentum = 0.9)))
methods.append(("Adagrad", adagrad(f, x0, eps, tk=0.01)))
methods.append(("Adam",adam(f, x0, eps, t = 0.02, beta1 = 0.8, beta2 = 0.99)))
''' 

seed(37)
eps = 10**-5
times = 50
x0 = [np.array(search[:, 0] + rand(len(search)) * (search[:, 1] - search[:, 0])) for i in range(times)]#np.array([3.0,4.0])
tsearch = np.arange(0.1,1,0.1)
msearch = np.arange(0.1,1,0.1)
msearch = np.arange(1,2,1)
its_t = []


gd_names = ['GD','SGD','NAG','AG','A']

K = 1

T0 = time.time()
for n in gd_names:
  min_its = []
  opt_t = []
  xk = []
  ticking = []
  
  if (n == 'NAG') or (n == 'GD'):
    msearch = np.arange(0.1,1,0.1)
    opt_m = []
  else:
    msearch = np.arange(1,2,1)
  
  for i in range(len(x0)): 
    adt = []
    for ti in tsearch:
      for mi in msearch:
        if n == "GD":
          #gd = grad_descent(f,x0[i],eps,t=ti)
          gd = grad_descent(f,x0[i],eps,"BT",a=ti,b=mi,s=1)
          if (sum(abs(gd[1][len(gd[1])-1]) > 1e10) > 0) or (isnan(sum(abs(gd[1][len(gd[1])-1])))):
            adt.append(10001)
          else:
            adt.append( gd[0] )
        if n == "SGD":
          sg = sgd(f,x0[i],eps,search,"Constant",batch = 1,t=ti,a=0.5,b=0.5,s=1)
          if (sum(abs(sg[1][len(sg[1])-1]) > 1e10) > 0) or (isnan(sum(abs(sg[1][len(sg[1])-1])))):
            adt.append(10001)
          else:
            adt.append( sg[0] )
        if n == "NAG":
          nag = nesterov(f,x0[i],eps,tk = ti, momentum = mi)
          if (sum(abs(nag[1][len(nag[1])-1]) > 1e10) > 0) or (isnan(sum(abs(nag[1][len(nag[1])-1])))):
            adt.append(10001)
          else:
            adt.append( nag[0] )
        if n == "AG":
          ag = adagrad(f, x0[i], eps, tk=ti)
          if (sum(abs(ag[1][len(ag[1])-1]) > 1e10) > 0) or (isnan(sum(abs(ag[1][len(ag[1])-1])))):
            adt.append(10001)
          else:
            adt.append( ag[0] )
        if n == "A":
          ad = adam(f, x0[i], eps, t = ti, beta1 = 0.8, beta2 = 0.99)
          if (sum(abs(ad[1][len(ad[1])-1]) > 1e10) > 0) or (isnan(sum(abs(ad[1][len(ad[1])-1])))):
            adt.append(10001)
          else:
            adt.append( ad[0] )
        
    ind = min(range(len(adt)), key=adt.__getitem__)
    tind = floor(ind/len(tsearch))
    tsrch = np.arange(tsearch[tind]-0.05,tsearch[tind]+0.05,0.01)
    
    if (n == 'NAG') or (n == 'GD'):
      mind = ind - tind*len(msearch)
      msrch = np.arange(msearch[mind]-0.05,msearch[mind]+0.05,0.01)
    else:
      msrch = np.arange(1,2,1)
      
    adt1 = []
    xs = [] 
    ticks = []
    for ti in tsrch:
      for mi in msrch:
        if n == "GD":
          t0 = time.time()
          #gd = grad_descent(f,x0[i],eps,t=ti)
          gd = grad_descent(f,x0[i],eps,"BT",a=ti,b=mi,s=1)
          t1 = time.time()
          if (sum(abs(gd[1][len(gd[1])-1]) > 1e10) > 0) or (isnan(sum(abs(gd[1][len(gd[1])-1])))):
            adt1.append(10001)
          else:
            adt1.append( gd[0] )
          xs.append( gd[1] )
          ticks.append(t1-t0)
        if n == "SGD":
          t0 = time.time()
          sg = sgd(f,x0[i],eps,search,"Constant",batch = 1,t=ti,a=0.5,b=0.5,s=1)
          t1 = time.time()
          if (sum(abs(sg[1][len(sg[1])-1]) > 1e10) > 0) or (isnan(sum(abs(sg[1][len(sg[1])-1])))):
            adt1.append(10001)
          else:
            adt1.append( sg[0] )
          xs.append( sg[1] )
          ticks.append(t1-t0)
        if n == "NAG":
          t0 = time.time()
          nag = nesterov(f,x0[i],eps,tk = ti, momentum = mi)
          t1 = time.time()
          if (sum(abs(nag[1][len(nag[1])-1]) > 1e10) > 0) or (isnan(sum(abs(nag[1][len(nag[1])-1])))):
            adt1.append(10001)
          else:
            adt1.append( nag[0] )
          xs.append( nag[1] )
          ticks.append(t1-t0)
        if n == "AG":
          t0 = time.time()
          ag = adagrad(f, x0[i], eps, tk=ti)
          t1 = time.time()
          if (sum(abs(ag[1][len(ag[1])-1]) > 1e10) > 0) or (isnan(sum(abs(ag[1][len(ag[1])-1])))):
            adt1.append(10001)
          else:
            adt1.append( ag[0] )
          xs.append( ag[1] )
          ticks.append(t1-t0)
        if n == "A":
          t0 = time.time()
          ad = adam(f, x0[i], eps, t = ti, beta1 = 0.9, beta2 = 0.999)
          t1 = time.time()
          if (sum(abs(ad[1][len(ad[1])-1]) > 1e10) > 0) or (isnan(sum(abs(ad[1][len(ad[1])-1])))):
            adt1.append(10001)
          else:
            adt1.append( ad[0] )
          xs.append( ad[1] )
          ticks.append(t1-t0)
        
        #print((gd_names.index(n)*len(tsrch)*len(msrch)+(list(tsrch).index(ti)+1)*(list(msrch).index(mi)+1))/(len(gd_names)*len(tsrch)*len(msrch))*100)  
          
    if not range(len(adt1)):
      min_its.append(100000)
      opt_t.append(0)
      ticking.append( 10000000000 )
    else:  
      ind = min(range(len(adt1)), key=adt1.__getitem__)
      if len(tsrch) == len(msrch):
        tind = floor(ind/len(tsrch))
      else:
        if len(tsrch) > len(msrch):
          tind = ceiling(ind/len(tsrch))
        else:
          tind = floor(ind/len(msrch))
      min_its.append(adt1[ind])
      opt_t.append(tsrch[tind])
      ticking.append( ticks[ind] )
      xk.append( xs[ind] )      
      if (n == 'NAG') or (n == 'GD'):
        mind = ind - tind*len(msrch)
        opt_m.append(msrch[mind])
        
    print(( K )/(len(gd_names)*len(x0))*100)  
    K = K + 1
  #print((gd_names.index(n)+1)/len(gd_names)*100)

  
  if (n == 'NAG') or (n == 'GD'):
    its_t_app = {n:{'t':opt_t,'m':opt_m,'its':min_its,'xk':xk,'run_time':ticking}}
    its_t.append(its_t_app)
  else:
    its_t_app = {n:{'t':opt_t,'its':min_its,'xk':xk,'run_time':ticking}}
    its_t.append(its_t_app)

T1 = time.time()
tt = T1-T0
tt/3600

for dic in its_t:
  for key in dic:
    print(key,dic[key]['its'],dic[key]['xk'][0][len(dic[key]['xk'][0])-1],dic[key]['run_time'])

f_out = its_t

newpath = '/home/kshitij/Desktop/College/SFU/Cources/923/Project/outputs/'+name+'/' 
if not os.path.exists(newpath):
  os.makedirs(newpath)

for n in gd_names: 
  gdi = gd_names.index(n)
  f_xk = f_out[gdi][n]['xk']
  if n == 'GD':
    f_its = pd.DataFrame([f_out[gdi][n]['its'],f_out[gdi][n]['run_time'],f_out[gdi][n]['t'],f_out[gdi][n]['m']]).T
    f_its = f_its.rename({0: 'its'+n, 1: 'run_time',2:'alpha',3:'beta'}, axis=1)
    f_its.to_csv('/home/kshitij/Desktop/College/SFU/Cources/923/Project/outputs/'+name+'/'+name+'_'+n+'_details.csv')
  else:
    if n == 'NAG':
      f_its = pd.DataFrame([f_out[gdi][n]['its'],f_out[gdi][n]['run_time'],f_out[gdi][n]['t'],f_out[gdi][n]['m']]).T
      f_its = f_its.rename({0: 'its'+n, 1: 'run_time',2:'t',3:'m'}, axis=1)
      f_its.to_csv('/home/kshitij/Desktop/College/SFU/Cources/923/Project/outputs/'+name+'/'+name+'_'+n+'_details.csv')
    else:
      f_its = pd.DataFrame([f_out[gdi][n]['its'],f_out[gdi][n]['run_time'],f_out[gdi][n]['t']]).T
      f_its = f_its.rename({0: 'its'+n, 1: 'run_time',2:'t'}, axis=1)
      f_its.to_csv('/home/kshitij/Desktop/College/SFU/Cources/923/Project/outputs/'+name+'/'+name+'_'+n+'_details.csv')
      
  for i in range(times):
    if i == 0:
      fxkt0 = np.array(f_xk[i])
      fxktdf0 = pd.DataFrame(fxkt0)
      fxktdf0 = fxktdf0.rename({0: 'x1', 1: 'x2'}, axis=1)
    else:
      fxkt = np.array(f_xk[i])
      fxktdf = pd.DataFrame(fxkt)
      fxktdf = fxktdf.rename({0: 'x1', 1: 'x2'}, axis=1)
      fxktdf0 = pd.concat([fxktdf0, fxktdf], axis=1)
    
  fxktdf0.to_csv('/home/kshitij/Desktop/College/SFU/Cources/923/Project/outputs/'+name+'/'+name+'_'+n+'_xk.csv')    

plt.scatter(opt_t, min_its)


fnames = ['beales','booth','goldsteinprice','thc','rhe']
fnms =  ['Beale', 'Booth', 'Goldstein Price', 'Three Humped Camel', 'Rotated Hyper Ellipsoid']

name = fnames[0]
pname = fnms[0]

seed(314)
f_all = getattr(fn,name)
f = f_all[0]
fmin = f(*f_all[1])
xmin, xmax, xstep = f_all[2][0][0], f_all[2][0][1], .2
ymin, ymax, ystep = f_all[2][1][0], f_all[2][1][1], .2
search = asarray([[xmin, xmax], [ymin, ymax]])

x0 = np.array(search[:, 0] + rand(1) * (search[:, 1] - search[:, 0]))#np.array([3.0,4.0])

for n in gd_names:
  data = pd.read_csv('/home/kshitij/Desktop/College/SFU/Cources/923/Project/outputs/'+name+'/'+name+'_'+n+'_xk.csv')
  if n == 'GD':
    #methods = [("Gradient",grad_descent(f,x0,eps,"BT",a=mean(data['alpha']),b=mean(data['beta']),s=1))]
    print(mean(data['alpha']),mean(data['beta']),mean(data['its'+n]))
  if n == 'SGD':
   # methods.append(("Stochastic Gradient", sgd(f,x0,eps,search,"Constant",batch = 1,t=mean(data['t']),a=0.5,b=0.5,s=1)))
    print(mean(data['t']),mean(data['its'+n]))
  if n== 'NAG':
  #  methods.append(("Nesterov",nesterov(f,x0,eps,tk = mean(data['t']), momentum = mean(data['m']))))
    print(mean(data['t']),mean(data['m']),mean(data['its'+n]))
  if n == 'AG':  
 #   methods.append(("Adagrad", adagrad(f, x0, eps, tk=mean(data['t']))))
    print(mean(data['t']),mean(data['its'+n]))
  if n == 'A':  
#    methods.append(("Adam",adam(f, x0, eps, t = mean(data['t']), beta1 = 0.9, beta2 = 0.999)))
    print(mean(data['t']),mean(data['its'+n]))


#x0 = np.array([29.0,59.0])

for fnm in range(len(fnames)):
  name = fnames[fnm]
  pname = fnms[fnm]

  f_all = getattr(fn,name)
  f = f_all[0]
  fmin = f(*f_all[1])

  for n in gd_names:
    
    data = pd.read_csv('/home/kshitij/Desktop/College/SFU/Cources/923/Project/outputs/'+name+'/'+name+'_'+n+'_xk.csv')
  
    fig = plt.figure(figsize =(10, 7))
    plt.axhline(y=0.0, color='k', linestyle='-')
  
    for i in range(int((len(list(data))-1)/2)):
      plt.plot(data['Unnamed: 0'],f(data[list(data)[(i+1)*2-1]],data[list(data)[(i+1)*2]])-fmin)
    
    plt.xlabel('Iterations')
    plt.ylabel('Function Value')
    plt.title(pname+" " + n)
    plt.savefig('/home/kshitij/Desktop/College/SFU/Cources/923/Project/outputs/'+name+'/'+name+'_'+n+'.png')

#  plt.ylim(0,1000)

# Creating plot
plt.boxplot(data['itsA'])


[print(methods[i][0]) for i in range(len(methods))] # Method names.
[print(methods[i][1][0]) for i in range(len(methods))] # number of iterations per method.

# Iterate paths 
#x0_ = x0.reshape(-1,1)
x0_ = np.transpose(x0)

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
plt.title("Rotated Hyper Ellipsoid")

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

