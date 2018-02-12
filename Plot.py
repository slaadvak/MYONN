#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import numpy

matplotlib.rcParams['figure.figsize']

def plot_record(inputs, label, percentage, prediction):
    #plt.figure()
    plt.imshow(numpy.reshape(inputs, (28,28)), cmap='Greys', interpolation='None')
    plt.title("Predicted: {} ({:.2f}%), Actual: {}".format(prediction, percentage, label), fontsize = 15)
    plt.show()
    