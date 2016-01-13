from __future__ import print_function
import sys
import numpy as np
from pylab import *
import json

def load(fn):
    with open(fn) as f:
        return np.array(json.load(f))

def load_labels():
    with open('mnist_labels.json') as f:
        return np.array(json.load(f))

def plot_latents(data, labels):
    colors = "r b y g m c brown pink grey gold".split(" ")
    axis('equal')
    xlim((-4, 4))
    ylim((-4, 4))
    gcf().set_size_inches(6, 6)
    for i in range(10):
        ix = np.where(labels == i)
        xy = data[ix]
        scatter(xy[:,0], xy[:,1], c=colors[i], marker="x", s=8, label=i)
    #legend()
    grid()

#print(sys.argv)
data = load(sys.argv[1])
labels = load_labels()[0:1000]
plot_latents(data, labels)
savefig(sys.argv[2], dpi=120)
