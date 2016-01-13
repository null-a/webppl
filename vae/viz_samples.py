from __future__ import print_function
import sys
import numpy as np
from pylab import *
import json

def load_samples(fn):
    with open(fn) as f:
        return np.array(json.load(f))

def plot_digits(digits, n):
    # Show n*n digits in a grid.
    # digits[i] is expected to be a a (flat) array of length 28*28=768.
    f, axarr = subplots(n, n)
    for i in range(n):
        for j in range(n):
            ax = axarr[i, j]
            ax.imshow(samples[i * n + j].reshape((28, 28)), interpolation='none', cmap='gray')
            setp(ax.get_xticklabels(), visible=False)
            setp(ax.get_yticklabels(), visible=False)

#print(sys.argv)
samples = load_samples(sys.argv[1])
plot_digits(samples, 4)
savefig(sys.argv[2])
