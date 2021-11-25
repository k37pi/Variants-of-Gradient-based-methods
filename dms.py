#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:23:12 2021

@author: kshitij
"""
from autograd import elementwise_grad, value_and_grad
import sys
sys.path.append('home/kshitij/Desktop/College/SFU/Cources/923/Project')
import functions as fn

from inspect import getmembers, isfunction


def norm(x1,x2):
    return np.sqrt(x1**2+x2**2)

#def grad_descent(g1,g2,eps,t,x0):
def grad_descent(f,x0,eps,lin_srch,t=0.9,a=0.5,b=0.5,s=1): 
  k = 0
  xk = [x0]
  
  grad = np.array((elementwise_grad(f, argnum=0)(*np.array(x0,dtype=float)),elementwise_grad(f, argnum=1)(*np.array(x0,dtype=float))),dtype = float)
  fun_val = f(*x0)
  while (norm(*grad) > eps) & (k < 10000):
    k = k + 1
    if lin_srch == "Constant":
      x0 = x0 - t*grad       
    if lin_srch == "BT":
      t = s
      while fun_val-f(*np.array(x0-t*grad,dtype=float)) < a*t*norm(*grad)**2:
        t = b*t
      x0 = x0 - t*grad       
      fun_val = f(*x0)
    grad = np.array((elementwise_grad(f, argnum=0)(*np.array(x0,dtype=float)),(elementwise_grad(f, argnum=1)(*np.array(x0,dtype=float)))),dtype = float) 
    xk.append(x0)
    print(k)
  return [k,xk]  


def sgd(f,x0,eps,search,lin_srch,batch = 1,t=0.9,a=0.5,b=0.5,s=1): 
  k = 0
  xk = [x0]
  grad = 0
  for i in range(batch):
    xj = np.array(search[:, 0] + rand(len(search)) * (search[:, 1] - search[:, 0]))
    grad = grad + np.array((elementwise_grad(f, argnum=0)(*np.array(xj,dtype=float)),elementwise_grad(f, argnum=1)(*np.array(xj,dtype=float))),dtype = float)
  grad = grad/batch 
  fun_val = f(*x0)

  while (norm(*grad) > eps) & (k < 100):
    k = k + 1
    if lin_srch == "Constant":
      x0 = x0 - t*grad       
    if lin_srch == "BT":
      t = s
      while fun_val-f(*np.array(x0-t*grad,dtype=float)) < a*t*norm(*grad)**2:
        t = b*t
      x0 = x0 - t*grad       
      fun_val = f(*x0)
    xk.append(x0)

    grad = 0
    for i in range(batch):
      xj = np.array(search[:, 0] + rand(len(search)) * (search[:, 1] - search[:, 0]))
      grad = grad + np.array((elementwise_grad(f, argnum=0)(*np.array(xj,dtype=float)),elementwise_grad(f, argnum=1)(*np.array(xj,dtype=float))),dtype = float)
    grad = grad/batch 
    print(k)
  return [k,xk]  


def nesterov(f, x0, eps, tk=0.01, momentum=0.8):
  xk = [x0]
  # no change in the beginning 
  change = 0.0
  x1 = x0 + momentum*change # what the next iterate could be
  grad = np.array((elementwise_grad(f, argnum=0)(*np.array(x1,dtype=float)),(elementwise_grad(f, argnum=1)(*np.array(x1,dtype=float)))),dtype = float)
  change = momentum*change - tk*grad
  x0 = x0 + change
  xk.append(x0)      
  k = 1
  while (norm(*((x1-x0))) > eps) & (k < 10000):  
    k = k + 1
    x1 = x0 + momentum*change # what the next iterate could be
    grad = np.array((elementwise_grad(f, argnum=0)(*np.array(x1,dtype=float)),(elementwise_grad(f, argnum=1)(*np.array(x1,dtype=float)))),dtype = float)
    change = momentum*change - tk*grad
    x0 = x0 + change
    xk.append(x0)      
  return [k,xk]

def adagrad(f, x0, eps, tk=0.01):
  xk = [x0]
  # 0 sum
  grad2_sum = 0
  k = 1
  grad = np.array((elementwise_grad(f, argnum=0)(*np.array(x0,dtype=float)),(elementwise_grad(f, argnum=1)(*np.array(x0,dtype=float)))),dtype = float)
  grad2_sum = grad2_sum + grad**2.0
  alpha = tk / (1e-8 + np.sqrt(grad2_sum))
  x0 = x0 - alpha * grad
  xk.append(x0)
    
  while (norm(*((xk[k]-xk[k-1]))) > eps) & (k < 10000): 
    k = k+1
    grad = np.array((elementwise_grad(f, argnum=0)(*np.array(x0,dtype=float)),(elementwise_grad(f, argnum=1)(*np.array(x0,dtype=float)))),dtype = float)
    grad2_sum = grad2_sum+ grad**2.0
    alpha = tk / (1e-8 + np.sqrt(grad2_sum))
    x0 = x0 - alpha * grad
    xk.append(x0)
    print(k)
  return [k,xk]

def adam(f, x0, eps, t = 0.02, beta1 = 0.8, beta2 = 0.9):
  xk = [x0]
  m = 0
  v = 0
  k = 1
  grad = np.array((elementwise_grad(f, argnum=0)(*np.array(x0,dtype=float)),(elementwise_grad(f, argnum=1)(*np.array(x0,dtype=float)))),dtype = float)
  m = beta1 * m + (1.0 - beta1) * grad
  v = beta2 * v + (1.0 - beta2) * grad**2
  mhat = m / (1.0 - beta1**(k))
  vhat = v / (1.0 - beta2**(k))
  x0 = x0 - t * mhat / (np.sqrt(vhat) + 1e-8)
  xk.append(x0)
  while (norm(*((xk[k]-xk[k-1]))) > eps) & (k < 10000): 
  #while (norm(*grad) > eps) & (k < 10000): 
    k = k+1
    grad = np.array((elementwise_grad(f, argnum=0)(*np.array(x0,dtype=float)),(elementwise_grad(f, argnum=1)(*np.array(x0,dtype=float)))),dtype = float)
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * grad**2
    mhat = m / (1.0 - beta1**(k))
    vhat = v / (1.0 - beta2**(k))
    x0 = x0 - t* mhat / (np.sqrt(vhat) + 1e-8)
    xk.append(x0)
    print(k)
  return [k,xk]
