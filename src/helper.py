"""
Created by kunal on 11/23/17
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_func(X, Y_gt, Y):
    fig = plt.figure(facecolor='white')
    plt.plot(range(12),X[0,...],'r')
    plt.plot(range(12), Y_gt[0, ...], 'g')
    plt.plot(range(12), Y[0, ...], 'b')
    plt.show()