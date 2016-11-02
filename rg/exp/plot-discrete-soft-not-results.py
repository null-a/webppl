import numpy as np
import matplotlib.pyplot as plt
import json
import math

def thin(run, n):
    new = []
    for i in range(len(run)):
        if i % n == 0:
            new.append(run[i])
    return new

def load(guide, length, run):
    fn = '-'.join([guide, length, run]) + '.json'
    path = 'rg/exp/discrete-soft-not-results/'
    with open(path + fn) as f:
        return thin(json.load(f), 500)

guides = 'rnn gru lstm'.split(' ')
lengths = [str(x) for x in [2, 4, 8, 16, 32]]
runs = [str(x) for x in range(3)]

def mean_and_var_of_runs(runs):
    # [run1, run2, run3, ...]
    # transpose
    # [[run1_0, run2_0, run3_0, ...], ...]
    # then compute mean and std dev of each element of list
    runsT = zip(*runs)
    means = []
    sds = []
    for i in xrange(len(runsT)):
        means.append(mean(runsT[i]))
        sds.append(sd(runsT[i]))
    return (means, sds)

def expectation(xs, f):
    return sum(f(x) for x in xs) / float(len(xs))

def mean(xs):
    return expectation(xs, lambda x: x)

def variance(xs):
    m = mean(xs)
    return expectation(xs, lambda x: (x - m) ** 2)

def sd(xs):
    return math.sqrt(variance(xs))

#fig = plt.gcf()
#fig.set_size_inches(18.5, 10.5)

plt.figure(figsize=(12,10))


ix = 0
for guide in guides:
    for length in lengths:
        ix += 1
        plt.subplot(len(guides), len(lengths), ix)
        plt.title('%s - %s' % (guide, length))
        plt.ylim(-2.5, -.5)
        if ((ix - 1) % len(lengths) == 0):
            plt.ylabel('elbo')
        c = 'b'


        for run in runs:
            data = load(guide, length, run)
            plt.plot(data, c=c)


        # allruns = [load(guide, reparam, length, run) for run in runs]
        # means, sds = mean_and_var_of_runs(allruns)
        # upper, lower = zip(*[(_mean + _sd, _mean - _sd) for (_mean, _sd) in zip(means, sds)])

        # plt.plot(means, c=c)
        # plt.fill_between(xrange(len(lower)), lower, upper, alpha=0.2, facecolor=c)

plt.suptitle('soft not dependency between 2 discrete variables (adam .001, 10 samples)')
plt.savefig('discrete-soft-not.png')
plt.show()
