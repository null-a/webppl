from __future__ import print_function
import sys
import numpy as np
from pylab import *
import json
import glob
import os.path

def load(fn):
    with open(fn) as f:
        return np.array(json.load(f))

def load_labels():
    with open('mnist_labels.json') as f:
        return np.array(json.load(f))

def plot_latents(data, labels):
    colors = "r b y g m c brown pink grey gold".split(" ")
    fig = gcf()
    fig.set_size_inches(6, 6)
    fig.clear()
    axis('equal')
    xlim((-4, 4))
    ylim((-4, 4))
    for i in range(10):
        ix = np.where(labels == i)
        xy = data[ix]
        scatter(xy[:,0], xy[:,1], c=colors[i], marker="x", s=8, label=i)
    #legend()
    grid()

#print(sys.argv)

labels = load_labels()[0:1000]

dirName = 'vae/latents'

for fn in glob.glob(os.path.join(dirName, '*.json')):
    pngfn = os.path.basename(fn).split('.')[0] + ".png";
    pngpath = os.path.join(dirName, pngfn)
    if not os.path.exists(pngpath):
        # Need to convert.
        data = load(fn)
        plot_latents(data, labels)
        savefig(pngpath, dpi=120)
