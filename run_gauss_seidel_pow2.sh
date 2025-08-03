#!/bin/sh
# Run Gauss_Seidel_C++ for all power-of-2 sizes between 2 and 500

EXE=Gauss_Seidel
SRC=Gauss_Seidel_C++.cpp

# Compile the program if not already compiled
c++ -O3 -o $EXE $SRC || { echo "Compilation failed"; exit 1; }

size=2
while [ $size -le 500 ]
do
  echo "==== Size: $size ===="
  for run in 1 2 3 4 5
  do
    echo "Run $run for size $size:"
    ./$EXE $size >> gauss_seidel_output_pow2.txt
  done
  echo ""
  size=`expr $size \* 2`
done
