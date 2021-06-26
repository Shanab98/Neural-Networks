# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 12:49:43 2021

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt
#%% 5.2.4   Steepest Descent for Convex Plane 
N = 100
s = 0.1
d = 0.01
x = np.zeros((N))
y = np.zeros((N))

#starting coordinates 
x[0] = 1 
y[0] = 1
for i in range(0,N-1): 
    xinc = -s*(4*((x[i]-1)**3) - 1)
    yinc = -s*2*(y[i]-2)
    
    if xinc**2 + yinc**2 < d**2:  
        x = x[0:i+1] #trim off excess zeros in initialized array
        y = y[0:i+1] #trim off excess zeros in initialized array
        break
    else: 
        x[i+1] = x[i] + xinc 
        y[i+1] = y[i] + yinc
    
plt.scatter(x,y)
plt.title('Steepest descent')
plt.show()
print('The Minimum is at: (' + str(x[i]) +', ' + str(y[i]) + ')')


#%% 5.2.5   Steepest Descent for Non Convex Plane
N = 1000 #increased iterations to 1000 so it could converge 
s = 0.0001 #decreased step size to avoid overshooting and missing the minima
d = 0.001 # decreased delta so that it didnt converge too easilly
x = np.zeros((N))
y = np.zeros((N))

#starting coordinates - change these until convergence at the lowest minima
x[0] = -1
y[0] = -2
for i in range(0,N-1): 
    xinc = -s*(4*x[i]**3 + 3*x[i]**2 - 12*x[i]) 
    yinc = -s*(4*y[i]**3 + 3*y[i]**2 - 12*y[i]) 
    
    if xinc**2 + yinc**2 < d**2:  
        x = x[0:i+1] #trim off excess zeros in initialized array
        y = y[0:i+1] #trim off excess zeros in initialized array
        break
    else: 
        x[i+1] = x[i] + xinc 
        y[i+1] = y[i] + yinc
    
plt.scatter(x,y)
plt.title('Steepest descent starting at: ( ' + str(x[0]) + ',' + str(y[0]) + ')' )
print('The Minimum is at: (' + str(x[i]) +', ' + str(y[i]) + ')')

# E 
print('number of iterations: ' , len(x))
cx = round(x[i], 1) #just for fun
cy = round(y[i], 1)#just for fun

x= np.arange(-3.2, 2.2, 0.1)
y= np.arange(-3.2, 2.2, 0.1)
X, Y = np.meshgrid(x, y)
rows, cols = X.shape

for r in range(rows): #just for fun
    for c in range(cols): 
        X[r,c] = round(X[r,c],1)
        Y[r,c] = round(Y[r,c],1)

z = (X**2 * (X-2) * (X+3)) + (Y**2 * (Y-2) * (Y+3))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Z = z.reshape(X.shape)

ax.plot_surface(X, Y, Z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

coords = np.argwhere((X==cx) & (Y==cy))[0]
print("Minima: " , Z[coords[0], coords[1]]) 