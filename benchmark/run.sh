#!/usr/bin/env bash
for i in {1..5}
do
    webppl benchmark/vae.wppl --require fs -- --N 50 --batchSize 50 --numSteps 10 --xDim 1    --hDim 1    --zDim 1
    webppl benchmark/vae.wppl --require fs -- --N 50 --batchSize 50 --numSteps 10 --xDim 3    --hDim 3    --zDim 3
    webppl benchmark/vae.wppl --require fs -- --N 50 --batchSize 50 --numSteps 10 --xDim 10   --hDim 10   --zDim 10
    webppl benchmark/vae.wppl --require fs -- --N 50 --batchSize 50 --numSteps 10 --xDim 30   --hDim 30   --zDim 30
    webppl benchmark/vae.wppl --require fs -- --N 50 --batchSize 50 --numSteps 10 --xDim 100  --hDim 100  --zDim 100
    webppl benchmark/vae.wppl --require fs -- --N 50 --batchSize 50 --numSteps 10 --xDim 300  --hDim 300  --zDim 300
    webppl benchmark/vae.wppl --require fs -- --N 50 --batchSize 50 --numSteps 10 --xDim 1000 --hDim 1000 --zDim 1000
done
