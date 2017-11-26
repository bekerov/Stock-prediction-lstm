"""
Created by kunal on 11/23/17
"""

import numpy as np
import matplotlib.pyplot as plt
from model import sequence_length

def plot_func(X, Y_gt, Y):
    fig = plt.figure(facecolor='white')
    plt.plot(range(sequence_length),X[0,...],'ro-')
    plt.plot(range(sequence_length), Y_gt[0, ...], 'go-')
    plt.plot(range(sequence_length), Y[0, ...], 'bo-')
    plt.legend(['Opening Price', 'Actual Closing Price', 'Predicted Closing Price'])
    plt.show()