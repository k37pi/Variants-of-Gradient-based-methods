#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:48:20 2021

@author: kshitij
"""
#%matplotlib qt5

import math
import sympy as sy
from sympy import evalf
import autograd.numpy as np
from numpy.linalg import eig

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation
from IPython.display import HTML

## many local minima
# 1 ackley 1 
a = 20
b=0.2
c=2*np.pi
ackley = [lambda x1,x2: -a*np.exp(-b*np.sqrt(0.5*(x1**2+x2**2))) -np.exp(0.5*(np.cos(c*x1)+np.cos(c*x2))) +a+np.exp(1), np.array([0.0,0.0]), [np.array([-32.768,32.768]),np.array([-32.768,32.768])]]

# 2 freud roth na
freud_roth = [lambda x, y: (-13+x+((5-y)*y-2)*y)**2 + (-29+x+((1+y)*y-14)*y)**2, [np.array([5.0,4.0]),np.array([(53.-4*np.sqrt(22))/3,(2.+np.sqrt(22))/3])],[np.array([-12.,12.]),np.array([-6.,8.])]]

# 3 Eggholder function 5
eggholder = [lambda x, y: -(y+47)*np.sin(np.sqrt(np.abs(y+0.5*x+47)))-y*np.sin(np.sqrt(np.abs(x-y-47))), np.array([512.0,404.2319]),[np.array([-512.0,512.0]),np.array([-512.0,512.0])]]

# 4 Griewank function 7
griewank = [lambda x, y: (x**2+y**2)/4000-np.cos(x)*np.cos(y/np.sqrt(2))+1, np.array([0.0,0.0]),[np.array([-600.0,600.0]),np.array([-600.0,600.0])]]

# 5 levy N13 11
levyn = [lambda x, y: np.sin(3*np.pi*x)**2+((x-1)**2)*(1+np.sin(3*np.pi*y)**2)+((y-1)**2)*(1+np.sin(2*np.pi*y)**2), np.array([1.0,1.0]),[np.array([-10.0,10.0]),np.array([-10.0,10.0])]]

# 6 Schwefel 15
schwefel = [lambda x, y: 837.9658 - x*np.sin(np.sqrt(np.abs(x))) - y*np.sin(np.sqrt(np.abs(y))), np.array([420.9687,420.9687]),[np.array([-500.0,500.0]),np.array([-500.0,500.0])]]

## bowl shaped
# 7 bohachevsky 17
bohachevsky = [lambda x, y: x**2 + 2*y**2 - 0.3*np.cos(3*np.pi*x) - 0.4*np.cos(4*np.pi*x) + 0.7, np.array([0.0,0.0]),[np.array([-100.0,100.0]),np.array([-100.0,100.0])]]

# 8 rotated hyper ellipsoid 19
rhe = [lambda x, y: 2*x**2 + y**2, np.array([0.0,0.0]),[np.array([-65.536,65.536]),np.array([-65.536,65.536])]]

# 9 rotated hyper ellipsoid 20
sphere = [lambda x, y: x**2 + y**2, np.array([0.0,0.0]),[np.array([-5.12,5.12]),np.array([-5.12,5.12])]]

# 10 sum of diff powers 21
sdp = [lambda x, y: np.abs(x) + y**2, np.array([0.0,0.0]),[np.array([-1.0,1.0]),np.array([-1.0,1.0])]]

# 11 trid 23
trid = [lambda x, y: (x-1)**2 + (y-1)**2 - x*y, np.array([2.0,2.0]),[np.array([-4.0,4.0]),np.array([-4.0,4.0])]]

## plate shaped
# 12 booth 24
booth = [lambda x, y: (x+2*y-7)**2 + (2*x+y-5)**2 - x*y, np.array([1.0,3.0]),[np.array([-10.0,10.0]),np.array([-10.0,10.0])]]

# 13 matyas 25
matyas = [lambda x, y: 0.26*(x**2 + y**2)-0.48*x*y, np.array([0.0,0.0]),[np.array([-10,10]),np.array([-10,10])]]

# 14 zakharov 27
zakharov = [lambda x, y: x**2 + y**2 + (0.5*x+y)**2 + (0.5*x+y)**4, np.array([0.0,0.0]),[np.array([-5.0,10.0]),np.array([-5.0,10.0])]]

## Valley shaped
# 15 3 humped camel 29
thc = [lambda x, y: 2*x**2 - 1.05*x**4 + x**6/6+ x*y + y**2, np.array([0.0,0.0]),[np.array([-5.0,5.0]),np.array([-5.0,5.0])]]

# 16 dixon price 31 
dixonprice = [lambda x, y: (x-1)**2 + 2*(2*y**2-x)**2, np.array([1.0,2**-0.5]),[np.array([-10.0,10.0]),np.array([-10.0,10.0])]]

# 17 rosenbrock 32 
rosenbrock = [lambda x, y: 100*(y**2-x)**2 + (x-1)**2, np.array([1.0,1.0]),[np.array([-5.0,10.0]),np.array([-5.0,10.0])]]

## other
# 18 beales 36
beales = [lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2, np.array([3.0,0.5]),[np.array([-4.5,4.5]),np.array([-4.5,4.5])]]

# 19 goldstein price 40
goldsteinprice = [lambda x, y: (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)), np.array([0.0,-1.0]),[np.array([-2.0,2.0]),np.array([-2.0,2.0])]]

# 20 styblinski tang 41
styblinskitang = [lambda x, y: 0.5*(x**4-16*x**2+5*x)+0.5*(y**4-16*y**2+5*y) , np.array([-2.903534,-2.903534]),[np.array([-5.0,5.0]),np.array([-5.0,5.0])]]

