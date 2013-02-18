#!/bin/bash
set -ex
cd src
gcc -o network train.c ann.c -lm
cd ..
mv src/network .
