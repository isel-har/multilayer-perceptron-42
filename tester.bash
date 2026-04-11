#!/bin/bash
count=1
while [ $count -le $1 ]
do
    python3 evaluation.py
    # python3 dist.py
    ./mlp split data_training.csv 10 > /dev/null 2>&1 && \
    ./mlp train config.json > /dev/null 2>&1 && \
    ./mlp predict data_test.csv
    echo "____________________________________________"
    ((count++))
done