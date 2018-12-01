
# coding: utf-8

# In[ ]:


import numpy as np
import math
import matplotlib.pyplot as plt

def say(i):
    print('hello')
class Pso():
    def __init__(self,dim,Xbound,Vbound,size,c1=2,c2=2,w=1,max_iter=1000):
        self.dim = dim
        self.Xbound = Xbound
        self.Vbound = Vbound
        self.size = size
        self.r1 = np.random.random(1)
        self.r2 = np.random.random(1)
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.X = np.zeros((self.size,self.dim))
        self.V = np.zeros((self.size,self.dim))
        self.pbestX = np.zeros((self.size,self.dim))
        self.gbestX = np.zeros((1,self.dim))
        self.pbesty = np.zeros(self.size)
        self.gbesty = 1e9
        self.y = 1e9
    def Y(self,X):
        self.y = np.sum(X[i]**2-math.cos(2*math.pi*X[i]) for i in range(self.dim))
        return self.y
    def init_individual(self):
        for xx in range(self.size):
            for yy in range(self.dim) :
                self.X[xx][yy] = (np.random.random(1) - 0.5) * 5.12
                self.V[xx][yy] = (np.random.random(1) - 0.5) * 0.512
            self.pbestX[xx] = self.X[xx]
            self.y = self.Y(self.X[xx])
            self.pbesty[xx] = self.y
            if self.y < self.gbesty:
                self.gbesty = self.y
                self.gbestX = self.X[xx]
        return self
    def iterator(self):
        yhistory = []
        for n in range(self.max_iter):
            for i in range(self.size):
                self.y = self.Y(self.X[i])
                if self.y < self.pbesty[i]:
                    self.pbesty[i] = self.y
                    self.pbestX[i] = self.X[i]
                    if y < self.gbesty:
                        self.gbesty = self.y
                        self.gbestX = X[i]
            for i in range(self.size):   
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbestX[i] - self.X[i]) +                             self.c2 * self.r2 * (self.gbestX - self.X[i])
                for j in range(self.dim):
                    if abs(self.V[i][j]) <= self.Vbound :
                        self.V[i][j] = self.Vbound
                self.X[i] = self.X[i] + self.V[i]
                for j in range(self.dim):
                    if abs(self.X[i][j]) <= self.Xbound :
                        self.X[i][j] = self.Xbound
            yhistory.append(self.gbesty)
        return self.gbesty
mypso = Pso(dim = 5,size = 5,Xbound = 5.12,Vbound = 0.512)
mypso.init_individual()
answer = mypso.iterator()
Xr = np.array([n for n in range(100)])
Yr = np.array(yhistory)
plt.plot(Xr,Yr)
plt.xlabel('n')
plt.ylabel('f(x)')
plt.show

