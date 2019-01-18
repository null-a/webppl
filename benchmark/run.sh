#!/usr/bin/env bash
for i in {1..5}
do
    # webppl benchmark/vae.wppl --require fs -- --N 50 --batchSize 50 --numSteps 10 --xDim 1    --hDim 1    --zDim 1
    # webppl benchmark/vae.wppl --require fs -- --N 50 --batchSize 50 --numSteps 10 --xDim 3    --hDim 3    --zDim 3
    # webppl benchmark/vae.wppl --require fs -- --N 50 --batchSize 50 --numSteps 10 --xDim 10   --hDim 10   --zDim 10
    # webppl benchmark/vae.wppl --require fs -- --N 50 --batchSize 50 --numSteps 10 --xDim 30   --hDim 30   --zDim 30
    # webppl benchmark/vae.wppl --require fs -- --N 50 --batchSize 50 --numSteps 10 --xDim 100  --hDim 100  --zDim 100
    # webppl benchmark/vae.wppl --require fs -- --N 50 --batchSize 50 --numSteps 10 --xDim 300  --hDim 300  --zDim 300
    # webppl benchmark/vae.wppl --require fs -- --N 50 --batchSize 50 --numSteps 10 --xDim 1000 --hDim 1000 --zDim 1000


    python3 benchmark/vae_pyro.py -N 50 --batch-size 50 --num-steps 10 --x-dim 1    --h-dim 1    --z-dim 1
    python3 benchmark/vae_pyro.py -N 50 --batch-size 50 --num-steps 10 --x-dim 3    --h-dim 3    --z-dim 3
    python3 benchmark/vae_pyro.py -N 50 --batch-size 50 --num-steps 10 --x-dim 10   --h-dim 10   --z-dim 10
    python3 benchmark/vae_pyro.py -N 50 --batch-size 50 --num-steps 10 --x-dim 30   --h-dim 30   --z-dim 30
    python3 benchmark/vae_pyro.py -N 50 --batch-size 50 --num-steps 10 --x-dim 100  --h-dim 100  --z-dim 100
    python3 benchmark/vae_pyro.py -N 50 --batch-size 50 --num-steps 10 --x-dim 300  --h-dim 300  --z-dim 300
    python3 benchmark/vae_pyro.py -N 50 --batch-size 50 --num-steps 10 --x-dim 1000 --h-dim 1000 --z-dim 1000

done
