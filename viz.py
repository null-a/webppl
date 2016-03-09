from __future__ import print_function

import sys
import os
import json

from pylab import *

if len(sys.argv) <= 1:
    print("Usage: python viz.py data_dir")
    exit()

# datadir is expected to be a directory containing one or more JSON
# files. The content of each file is expected to contain an array
# representing the history of a particular training run, serialized as
# JSON.

datadir = sys.argv[1]
jsonFiles = filter(lambda f: f.endswith(".json"), os.listdir(datadir))

colors = "r b y g m c brown pink grey gold".split(" ")

def load_data():
    data = {}
    for fn in jsonFiles:
        ix = fn.rindex("_")
        base = fn[0:ix]
        timestamp = fn[ix+1:-5] # strip extension
        if not base in data:
            data[base] = []
        with open(os.path.join(datadir, fn)) as f:
            data[base].append(json.load(f))
    return data

def plot_data(data):
    for i, config in enumerate(data):
        for j, run in enumerate(data[config]):
            plot(map(lambda x: -x, run), c=colors[i % len(colors)], label=config if j == 0 else None)
    title(datadir)
    xlabel('step')
    legend()
    grid()
    show()

data = load_data()
plot_data(data)
