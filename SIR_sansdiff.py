#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:08:56 2020

@author: cyril
"""
import matplotlib.pyplot as plt
import numpy as np

N=1000  #effectif de la population
s0 = N   #nombre de susceptible initial
i0 = 1  #nombre d'infectés initial
r0 = 0  #nombre de guéris ou morts initial

beta = 5  #coefficient de transmission
gamma = 1  #coefficient de "guérison"

dt = .01
T = 10   #durée de la simulation
Nt = int(T/dt)  #nombre d'itération

u = [s0,i0,r0]
Uplt = np.zeros((Nt,3))   #pour le plot
Uplt[0]=u

def zero_par_newton(g,x0,epsilon=1e-6,h=1e-4):
    x = x0
    while abs(g(x)) > epsilon:
            derivee = (g(x+h) - g(x)) / h
            x = x - g(x)/derivee
    return x

for t in range(1,Nt):
    def f_i(x):
        return x*(1-gamma*dt*(beta*u[0]/(gamma*N*(1+beta*dt*x/N)) - 1))-u[1]
    x0 = u[1]*(1+dt*(beta*u[0]/N - gamma))    #solution initiale par Euler explicite
    u[1] = zero_par_newton(f_i,x0,epsilon=1e-6,h=1e-4)
    u[0]=u[0]/(1+beta*dt*u[1]/N)
    u[2]=N-(u[0]+u[1])  #condition de conservation du nbre total
    Uplt[t]=u

tps = np.linspace(0,T,Nt)
plt.plot(tps,Uplt[:,0], label="Susceptibles")
plt.plot(tps,Uplt[:,1], label="Infectés")
plt.plot(tps,Uplt[:,2], label="Guéris ou morts")
plt.legend()

plt.show()