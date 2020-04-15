#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:04:45 2020

@author: cyril
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from tqdm import tqdm
from celluloid import Camera

N=1000  #effectif de la population
s0 = N   #nombre de susceptible initial
i0 = 1  #nombre d'infectés initial
r0 = 0  #nombre de guéris ou morts initial

beta = 10  #coefficient de transmission
gamma = 1  #coefficient de "guérison"
coef_diff = [0,0,0]     #coefficient de diffusion (respectivement de s,i et r)

dt = .01
T = 10   #durée de la simulation
Nt = int(T/dt)  #nombre d'itération

dx = .01
L=1
Nx=int(L/dx)

u = np.zeros(3*Nx)  #répartition initiale de SIR
for x in range(Nx):
    u[x] = N*np.exp(-0.01*(x-Nx/2)**2)  #répartition des susceptibles
    u[x+Nx] = 50   #répartition des infectés
    u[x+2*Nx] = 0    #répartition des guéris/morts

Uplt = np.zeros((Nt,3*Nx))   #pour le plot
for x in range(3*Nx) : 
    Uplt[0,x]=u[x]
    
#############Construction des matrices##############
B = -2*np.eye(Nx)
for i in range(Nx-1) : 
    B[i,i+1] = 1
    B[i+1,i] = 1
    
B[0,Nx-1]=1
B[Nx-1,0]=1

A = np.zeros((3*Nx,3*Nx))
D = np.zeros((3*Nx,3*Nx))

for k in range(3) : 
    A[k*Nx:(k+1)*Nx,k*Nx:(k+1)*Nx]=B
    D[k*Nx:(k+1)*Nx,k*Nx:(k+1)*Nx]=coef_diff[k]*np.eye(Nx)

A=(1/dx**2)*A   #matrice laplacien
M=D.dot(A)

Mi = np.linalg.inv(np.eye(3*Nx)-dt*M)

#############Définition de la source##############
def f(uv):
    tab=np.zeros(3*Nx)
    for x in range(Nx) : 
        tab[x] = -beta*uv[x]*uv[x+Nx]/N
        tab[x+Nx] = beta*uv[x]*uv[x+Nx]/N -gamma*uv[x+Nx]
        tab[x+2*Nx]= gamma*uv[x+Nx]
    return tab

#############Schéma numérique##############
for t in range(1,Nt):
    u=u + dt*(M.dot(u)+f(u))    #explicite
    #u = Mi.dot(u+dt*f(u))       #implicite
    Uplt[t]=u

#############Plotting##############

# esp = np.linspace(0,L,Nx)
# for t in range(Nt):
#     plt.figure(figsize=(11, 3))
#     plt.subplot(131)
#     plt.title('Susceptibles')
#     plt.axis([0, L, 0, N])
#     plt.plot(esp,Uplt[t,0:Nx])
#     plt.subplot(132)
#     plt.title('Infectés')
#     plt.axis([0, L, 0, N])
#     plt.plot(esp,Uplt[t,Nx:2*Nx])
#     plt.subplot(133)
#     plt.title('Guéris ou morts')
#     plt.axis([0, L, 0, N])
#     plt.plot(esp,Uplt[t,2*Nx:3*Nx])
#     plt.pause(0.1) # pause avec duree en seconde
    
# plt.show()
