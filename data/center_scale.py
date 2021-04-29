#!/usr/bin/env python

from sklearn import preprocessing
from numpy import genfromtxt
import numpy as np


X = genfromtxt('data/wine_2.csv', delimiter=',')
Y = genfromtxt('data/wine_2_label.csv', delimiter=',')

X_scaled = preprocessing.scale(X)

print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

