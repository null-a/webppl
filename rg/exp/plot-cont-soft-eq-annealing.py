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

def load(guide, reparam, length, run):
    fn = '-'.join([guide, 'rand', 'none', 'anneal', 'true', length, run]) + '.json'
    path = 'rg/exp/cont-soft-eq-annealing-results/'
    with open(path + fn) as f:
        return thin(json.load(f), 500)

guides = 'rnn gru lstm'.split(' ')
reparams = 'true false'.split(' ')
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
        plt.ylim(-5, -2)
        if ((ix - 1) % len(lengths) == 0):
            plt.ylabel('elbo')
        # for reparam in reparams:
        #     c = 'r' if reparam == 'true' else 'b'


        for run in runs:
            data = load(guide, True, length, run)
            plt.plot(data, c='r')


            # allruns = [load(guide, reparam, length, run) for run in runs]
            # means, sds = mean_and_var_of_runs(allruns)
            # upper, lower = zip(*[(_mean + _sd, _mean - _sd) for (_mean, _sd) in zip(means, sds)])

            # plt.plot(means, c=c)
            # plt.fill_between(xrange(len(lower)), lower, upper, alpha=0.2, facecolor=c)

plt.suptitle('soft eq dependency between 2 continuous variables\n(adam .001, 10 samples, factor score annealed over first 10k steps)')
plt.savefig('cont-soft-eq-annealed.png')
plt.show()
