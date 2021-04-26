# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 09:19:33 2020

@author: keray
"""

import numpy as np
import matplotlib.pyplot as plt
from random import randrange

class Perceptron:
    
    def __init__(self, dimension, max_iter, learning_rate):
        self.dimension = dimension
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        
        
    def fit(self,X,y):
        if(np.dot(x,self.frontiere[1:])+self.frontiere[0]<=0):
            return -1
        else:
            return 1
    
    def predict(self,x):
        for _ in range(self.max_iter):
            for _ in range(len(y)):
                i=randrange(0,len(y))
                if(self.predict(X[i])==-1):
                    if y[i]==1:
                        self.frontiere[1:]=self.frontiere[1:]+self.learning_rate*X[i]
                        self.frontiere[0]=self.frontiere[0]+self.learning_rate
                if(self.predict(X[i])==1):
                    if y[i]==-1:
                        self.frontiere[1:]=self.frontiere[1:]-self.learning_rate*X[i]
                        self.frontiere[0]=self.frontiere[0]-self.learning_rate
                    
