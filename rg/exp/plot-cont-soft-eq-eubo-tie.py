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

def load(n, length, tie, run):
    fn = '-'.join(['rnn', 'n', n, 'len', length, 'tie', tie, run]) + '.json'
    path = 'rg/exp/cont-soft-eq-eubo-results/'
    with open(path + fn) as f:
        return thin(json.load(f), 50)



numhids = [str(x) for x in [4, 8, 16, 32]]
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
for n in numhids:
    for length in lengths:
        ix += 1
        plt.subplot(len(numhids), len(lengths), ix)
        plt.title('hid=%s - len=%s' % (n, length))
        plt.ylim(-3, 0)

        if (ix-1 < (len(numhids)-1) * len(lengths)):
            #frame=plt.gca()
            #frame.axes.get_xaxis().set_visible(False)
            plt.gca().set_xticklabels([''] * 80)

        if ((ix - 1) % len(lengths) == 0):
            plt.ylabel('eubo')
        else:
            plt.gca().set_yticklabels([''] * 10)

        for tie in ['true', 'false']:
            c = 'darkorange' if tie == 'true' else 'purple'


            for run in runs:
                data = load(n, length, tie, run)
                plt.plot(data, c=c)


            # allruns = [load(guide, reparam, length, run) for run in runs]
            # means, sds = mean_and_var_of_runs(allruns)
            # upper, lower = zip(*[(_mean + _sd, _mean - _sd) for (_mean, _sd) in zip(means, sds)])

            # plt.plot(means, c=c)
            # plt.fill_between(xrange(len(lower)), lower, upper, alpha=0.2, facecolor=c)

plt.suptitle('soft eq dependency between 2 continuous variables\n'
             + '(adam .001, ~10k examples traces, mini batch size=50, steps=4000)\n'
             + 'orange=rnn, purple=no weight sharing in update net, '
             + 'hid=size of rnn state, len=size of model')
plt.savefig('cont-soft-eq-eubo.png')
plt.show()
