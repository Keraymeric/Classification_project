# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:30:19 2020

@author: keray
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from perceptron import Perceptron

df= pd.read_csv("iris.csv")

perceptron = Perceptron(dimension=2, max_iter=100, learning_rate=0.1)

X_data= df.iloc[0:100,[0,2]].values

y_data= df.iloc[0:100,4].values
                
y_data= np.where(y_data == "setosa",-1,1)

colors = {-1:'red',1:'blue'}
y_colors = [colors[y] for y in y_data]

plt.scatters(X_data[:,0], X_data[:,1], c=y_colors, s=100)

plt.scatter(X_data[:,0], X_data[:,1],c=y_colors, s=100)
plt.show()