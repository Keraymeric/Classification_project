# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 09:25:34 2020

@author: keray
"""

import numpy as np

from perceptron import Perceptron

X_train=[]
X_train.append(np.array([1,1]))
X_train.append(np.array([1,0]))
X_train.append(np.array([0,1]))
X_train.append(np.array([0,0]))
y_train = np.array([1,-1,-1,-1])

perceptron = Perceptron(dimension=2, max_iter=100, learning_rate=0.1)
perceptron.fit(X_train, y_train)

new_x= np.array([1,1])
print(perceptron.predict(new_x))

new_x=np.array([0,1])
print(perceptron.predict(new_x))

