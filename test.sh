#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
if [ $# -gt 0 ]; then
  mpirun -np $1 ./test
else
  ./test
fi
