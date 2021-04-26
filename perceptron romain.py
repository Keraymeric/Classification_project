# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from random import randrange

class Perceptron(object):
    
    def __init__(self, dimension, max_iter, learning_rate):
        self.frontiere=np.zeros(dimension+1)
        self.max_iter=max_iter
        self.learning_rate=learning_rate
                
    def predict(self,x):
        if(np.dot(x,self.frontiere[1:])+self.frontiere[0]<=0):
            return -1
        else:
            return 1

    def fit(self,X,y):
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
                    
