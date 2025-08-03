#!/bin/env zsh
# Run Gauss_Seidel_C++ for sizes 5 to 500 (step 5), 5 times each

EXE=Gauss_Seidel
SRC=Gauss_Seidel_C++.cpp

# Compile the program if not already compiled
g++ -O3 -o $EXE $SRC || { echo "Compilation failed"; exit 1; }

for n in {3..500..10}
do
  echo "==== Size: $n ===="
  for run in {1..5}
  do
    echo "Run $run for size $n:"
    ./$EXE $n >> gauss_seidel_output.txt
  done
  echo ""
done
