#!/bin/bash

NUMPY_PATH='/usr/lib/python3/dist-packages/numpy/core/include/numpy'
PYTHON_PATH='/usr/include/python3.10/'

g++ -O3 -march=corei7 '-mavx2' -std=c++17 -Wall -Wextra \
    -Wno-deprecated-copy -Wno-deprecated-declarations \
    main.cpp src/*.cpp \
    -I lib -I lib/eigen -I include \
    -I $NUMPY_PATH -I $PYTHON_PATH \
    -lpython3.10 \
    -o mlp
