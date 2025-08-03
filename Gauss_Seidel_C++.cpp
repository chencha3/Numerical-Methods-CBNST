// C++ Program to Implement Gauss Seidel Method

#include <unistd.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;
#define EPSILON 0.000000001 // till 3 correct decimal places

// Helper function
float findSum(int i, std::vector<vector<float>> a, int n) {
  float sum = 0;
  for (int j = 0; j < n; j++) {
    if (i != j)
      sum += a[i][j];
  }
  return sum;
}

// checks if Gauss Seidel Method is applicable and return true if yes otherwise
// return false
bool isMethodApplicable(vector<vector<float>> a, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (fabs(a[i][i]) > findSum(i, a, n))
        break;
      else
        return false;
    }
  }
  return true;
}

// prints the Value of Unknowns
void print(int iteration, float values[], int n) {
#if 0
  cout << "Iteration " << iteration << ": ";
  for (int i = 0; i < n; i++)
    cout << "value[" << i + 1 << "]= " << values[i] << " ";
  cout << endl;
#endif
}

void findValues(vector<vector<float>> a, int maxIterations, float values_old[],
                int n) {
  int i, j, k, iteration, flag = 0;
  float ratio, sum = 0;
  float values_new[n];
  // This loop is added for Gauss seidel ( not present in Jacobi method)
  for (int i = 0; i < n; i++)
    values_new[i] = 0;

  for (iteration = 1; iteration <= maxIterations; iteration++) {
    for (i = 0; i < n; i++) {
      sum = 0;
      for (j = 0; j < n; j++) {
        if (i != j)
          sum += a[i][j] *
                 values_new[j]; // in Gauss Jacobi, we took values_old[] here
      }

      values_new[i] = (a[i][n] - sum) / a[i][i];
    }
    // Now you have found the values of n unknowns for above iteration
    // Now check if your matching criteria satisfied , comparing with previous
    // iteration values
    for (k = 0; k < n; k++) {
      if (fabs(values_old[k] - values_new[k]) < EPSILON) {
        continue;
      } else {
        flag = 1;
        break;
      }
    }
    if (flag == 0) {
      // print final values of unknowns and return
      print(iteration, values_new, n);
      // cout << "Return with accurate answer with niters = " << iteration <<
      // "\n"; return;
      break;
    }
    flag = 0; // resetting the flag

    print(iteration, values_new, n); // To print intermediate roots

    // copy new values of unknowns to old value array
    for (k = 0; k < n; k++)
      values_old[k] = values_new[k];
  } // end of iteration loop
  print(iteration, values_new, n);
  cout << "Tot number of iterations: " << iteration << endl;

} // end of findValues()

void generateMatrix(vector<vector<float>> &a, int n) {
  srand(
      static_cast<unsigned int>(time(0)) ^
      static_cast<unsigned int>(getpid())); // Seed for random number generation
  for (int i = 0; i < n; i++) {
    float rowSum = 0;
    for (int j = 0; j < n; j++) {
      if (i != j) {
        // Random float in [-10, 10]
        a[i][j] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        rowSum += fabs(a[i][j]);
      }
    }
    // Diagonal element: strictly greater than sum of off-diagonal elements
    a[i][i] = rowSum + static_cast<float>(rand()) / RAND_MAX * 5.0f + 1.0f;
    // Random constant for augmented column
    a[i][n] = static_cast<float>(rand()) / RAND_MAX * 10.0f + 1.0f;
  }
}

int main(int argc, char *argv[]) {
  int i, j, k, x, y, maxIterations, n;
  float ratio;
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " <n> [maxIterations]\n";
    return 1;
  }
  n = atoi(argv[1]);
  if (argc >= 3)
    maxIterations = atoi(argv[2]);
  else
    maxIterations = 10000;
  vector<vector<float>> a(500, vector<float>(501));
  float values[500];

  generateMatrix(a, n);
  cout << "\nThe Augmented Matrix (" << n << "x" << n << ")\n";
  // Print the augmented matrix with aligned columns
  cout << fixed;
  cout.precision(5);
  int width = 12;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cout << setw(width) << a[i][j];
    }
    cout << endl;
  }

  if (!isMethodApplicable(a, n)) {
    cout << "\nGauss Seidel Method can't be applied";
    return 0;
  }
  // cout << "\nGauss Seidel Method is applicable\n\n";
  for (int i = 0; i < n; i++)
    values[i] = 0;
  findValues(a, maxIterations, values, n);
  return 0;
}
