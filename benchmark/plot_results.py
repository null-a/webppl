import glob
import json

from matplotlib import pyplot as plt

def mean(xs):
    assert len(xs) > 0
    result = 0
    for x in xs:
        result += x
    return result / float(len(xs))

def load(fn):
    with open(fn, 'r') as f:
        return json.loads(f.read())

keys = ['N', 'adBackend', 'batchSize', 'hDim', 'numSteps', 'stepSize', 'xDim', 'zDim']

def result_key(result_dict):
    condition = result_dict['condition']
    # keys = sorted(condition.keys())
    assert len(condition) == len(keys)
    return tuple(condition[key] for key in keys)

raw_results = [load(fn) for fn in glob.glob('./benchmark/results/*.json')]

# Group results by condition.
grouped_results = {}
for r in raw_results:
    key = result_key(r)
    if key not in grouped_results:
        grouped_results[key] = []
    grouped_results[key].append(r)

# num params vs. time, grouped by backend

results = {}

# results = {'adnn' : [ (num_params, avg_time),  ... ],
#            ... }

for key, results_for_group in grouped_results.items():
    num_params = results_for_group[0]['numParams']
    condition = results_for_group[0]['condition']
    backend = condition['adBackend']
    mean_elapsed = mean([result['elapsed'] for result in results_for_group])

    if backend not in results:
        results[backend] = []

    results[backend].append((num_params, mean_elapsed))

for backend, pairs in results.items():
    xs, ys = zip(*sorted(pairs))
    plt.plot(xs, [y/1000.0 for y in ys], label=backend, marker='.')

plt.title('VAE (num_steps=10, batch size=50, single hidden layer)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('total number of scalar parameters')
plt.ylabel('elapsed (secs)')
plt.legend()
plt.show()
