import glob
import json
from pprint import pprint as pp

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

# Flatten to list of pairs
grouped_results = [(k,v) for (k,v) in grouped_results.items()]

for group in grouped_results:
    results = group[1]
    print('------------------------------')
    pp(results[0]['condition'])
    elapsed_list = [result['elapsed'] for result in results]
    print('mean runtime (ms): %s' % mean(elapsed_list))
    print('  %s' % elapsed_list)
