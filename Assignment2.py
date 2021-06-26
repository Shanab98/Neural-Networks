# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 20:16:24 2021

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import random


#%% 4.1.2
"""
Consider two cubic clusters of points in R3
The first one is uniformly distributed in [14, 20] × [0, 6] × [0, 6] 
and the second in                         [0, 6] × [14, 20] × [14, 20]. 

Design a basic perceptron to separate the two clusters of points.
(a) Find the equation of the ideal separating plane.
(c) From part (a) deduce the ideal parameter values.
(d) Simulate a total of 4000 points in the two clusters using Python.
(e) Use M = 100 points to train the basic perceptron.
(f) Use another 1000 points to find the error rate.
(g) If the error rate is smaller than 0.1 stop. Otherwise increase M, and repeat parts (e)
"""
print("ideal separating plane: x1-x2-x3+10 =0") #calculated on paper 
print("Ideal values: w1 = 1, w2 = -1, w3 = -1, b = 10")

random.seed(10)

# data generation
N=2000
c1_coords = [14, 20, 0, 6, 0, 6]
c2_coords = [0, 6, 14, 20, 14, 20]

c1x1 = np.random.uniform(c1_coords[0], c1_coords[1], N) #x
c1x2 = np.random.uniform(c1_coords[2], c1_coords[3], N) #y
c1x3 = np.random.uniform(c1_coords[4], c1_coords[5], N) #z

c2x1 = np.random.uniform(c2_coords[0], c2_coords[1], N) #x
c2x2 = np.random.uniform(c2_coords[2], c2_coords[3], N) #y
c2x3 = np.random.uniform(c2_coords[4], c2_coords[5], N) #z

x = [*c1x1, *c2x1]
y = [*c1x2, *c2x2]
z = [*c1x3, *c2x3]
t = np.zeros((2*N,1))
t[0:N] = 0
t[N:2*N] = 1

fig =plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)
#ax.scatter(c1x1, c1x2, c1x3)
#ax.scatter(c2x1, c2x2, c2x3)

# parameter initialization 
w11 = 0.5-np.random.rand()
w12 = 0.5-np.random.rand()
w13 = 0.5-np.random.rand()
b1 = 0.5-np.random.rand()

M=100
data_indx = range(0,2*N)
sp = random.sample(list(data_indx), M+1000)  #sample - M for training, 1000 for error rate
# --------------------------------- Training ------------------------------- #
for i in range(0,M): 
    y1 = b1 + w11*x[sp[i]] + w12*y[sp[i]] + w13*z[sp[i]]
    if y1<0: 
        y1 = 0
    else: 
        y1= 1
        
    e = t[sp[i]] - y1
    w11 = w11 + e*x[sp[i]]
    w12 = w12 + e*y[sp[i]]
    w13 = w13 + e*z[sp[i]]
    b1 = b1 + e
    
# --------------------------------- Testing ------------------------------- #
er = 0
for i in range(M, M+1000): 
    y1 = b1 + w11*x[sp[i]] + w12*y[sp[i]] + w13*z[sp[i]]
    if y1<0: 
        y1 = 0
    else: 
        y1= 1
        
    e1 = abs(t[sp[i]] - y1)
    if e1 == 1: 
        er = er + 1

er = er/1000 
print("Error: ", er)


# Plotting 
xx = np.linspace(0,20,20)
yy = np.linspace(0,20,20)

X,Y = np.meshgrid(xx,yy)
Z = (-b1 - w11*X  - w12*Y)/w13

fig =plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z)
ax.scatter(x,y,z)
plt.title('Acheived Separation')
plt.show()
print('Calculated weighte: ',w11, w12, w13)
print('Calculated bias: ', b1)
w11 = 1
w12 = -1 
w13 = -1
b1=10
xx = np.linspace(0,20,20)
yy = np.linspace(0,20,20)


X,Y = np.meshgrid(xx,yy)
Z = (-b1 - w11*X  - w12*Y)/w13

fig =plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z)
ax.scatter(x,y,z)
plt.title("Ideal Separation")
plt.show()

#%% 4.2.1
# design a neural network that classifies 4 clusters 


N=1000 # data points for each cluster 
M = 1000 # number of training samples
T= 1000 # number of testing samples
p = [8,4,18,10,12,20,2,14]

x = np.zeros((4*N,2))
t = np.zeros((4*N,2))
r = np.random.normal(0,1.5,4*N)
the = 2*np.pi*np.random.rand(4*N)

# creating 4 clusters 
coord = 0
for n in range(0,4*N,N): 
    x[n:n+N,0] = p[coord] + r[n:n+N]*np.cos(the[n:n+N])
    coord= coord + 1
    x[n:n+N,1] = p[coord] + r[n:n+N]*np.sin(the[n:n+N])
    coord = coord + 1
n = 0
t[n:n+N,0] = 0
t[n:n+N,1] = 1

n = 1000
t[n:n+N,0] = 1
t[n:n+N,1] = 1

n = 2000
t[n:n+N,0] = 1
t[n:n+N,1] = 0

n = 3000
t[n:n+N,0] = 0
t[n:n+N,1] = 0

# parameter initialization 
w11 = 0.5-np.random.rand()
w12 = 0.5-np.random.rand()
b1 = 0.5-np.random.rand()

w21 = 0.5-np.random.rand()
w22 = 0.5-np.random.rand()
b2 = 0.5-np.random.rand()

data_indx = range(0,4*N)
sp = random.sample(list(data_indx), M+T)  #sample - M for training, 1000 for error rate
# --------------------------------- Training ------------------------------- #
for i in range(0,M): 
    y1 = b1 + w11*x[sp[i],0] + w12*x[sp[i],1] 
    if y1<0: 
        y1 = 0
    else: 
        y1= 1
        
    e = t[sp[i],0] - y1
    w11 = w11 + e*x[sp[i],0]
    w12 = w12 + e*x[sp[i],1]
    b1 = b1 + e
    
    y2 = b2 + w21*x[sp[i],0] + w22*x[sp[i],1] 
    if y2<0: 
        y2 = 0
    else: 
        y2= 1
        
    e = t[sp[i],1] - y2
    w21 = w21 + e*x[sp[i],0]
    w22 = w22 + e*x[sp[i],1]
    b2 = b2 + e
# --------------------------------- Testing ------------------------------- #
er = 0
for i in range(M, M+T): 
    y1 = b1 + w11*x[sp[i],0] + w12*x[sp[i],1]
    y2 = b2 + w21*x[sp[i],0] + w22*x[sp[i],1]

    if y1<0: 
        y1 = 0
    else: 
        y1= 1
        
    e1 = abs(t[sp[i]] - y1)
    
    if y2<0: 
        y2 = 0
    else: 
        y2= 1
        
    e1 = abs(t[sp[i],0] - y1)
    e2 = abs(t[sp[i],1] - y2)

    if e1 == 1 or e2 == 1: 
        er = er + 1

er = er/T
print("Error: ", er)

# Plotting 
plt.scatter(x[:,0], x[:,1])
xx =  np.linspace(0,20,20)
yy1 = (-b1 -w11*xx)/w12
yy2 = (-b2-w21*xx)/w22
plt.plot(xx,yy1)
plt.plot(xx,yy2)
plt.ylim(-5,30)
plt.show()

