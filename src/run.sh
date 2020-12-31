#! /bin/sh

./train.py -fold 0 -model rf
./train.py -fold 1 -model rf
./train.py -fold 2 -model rf
./train.py -fold 3 -model rf
./train.py -fold 4 -model rf
