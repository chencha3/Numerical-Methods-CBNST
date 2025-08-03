//C++ Program to Implement Gauss Seidel Method

#include<iostream>
#include<vector>
#include<cmath>
using namespace std;
#define EPSILON 0.001  // till 3 correct decimal places


//Helper function
float findSum(int i, std::vector<vector<float>> a,int n)
{
     float sum=0;
     for(int j=0;j<n;j++)
     {
       if(i!=j)
         sum+=a[i][j];
     }
     return sum;
}

//checks if Gauss Seidel Method is applicable and return true if yes otherwise return false
bool isMethodApplicable(vector<vector<float>> a,int n)
{
   for(int i=0;i<n;i++) {
       for(int j=0;j<n;j++) {
           if(fabs(a[i][i])>findSum(i,a,n))
              break;
           else
              return false;
       }
   }
   return true;
}

//prints the Value of Unknowns
void print(int iteration, float values[],int n)
{
  cout<<"Iteration "<<iteration<< ": ";
//   for(int i=0;i<n;i++)
//     cout<<"value["<<i+1<<"]= "<<values[i]<<" ";
  cout<<endl;
}


void findValues(vector<vector<float>> a, int maxIterations, float values_old[],int n)
{
    int i,j,k,iteration,flag=0;
    float ratio,sum=0;
    float values_new[n];
    //This loop is added for Gauss seidel ( not present in Jacobi method)
    for(int i=0;i<n;i++)
     values_new[i]=0;

    for(iteration=1;iteration<=maxIterations;iteration++)
    {
        for(i=0;i<n;i++)
        {
            sum=0;
            for(j=0;j<n;j++)
            {
                if(i!=j)
                sum+=a[i][j]*values_new[j];   // in Gauss Jacobi, we took values_old[] here
            }

            values_new[i]=(a[i][n] - sum)/a[i][i];
        }
        //Now you have found the values of n unknowns for above iteration
        //Now check if your matching criteria satisfied , comparing with previous iteration values
        for(k=0;k<n;k++)
        {
            if(fabs(values_old[k]-values_new[k])<EPSILON)
                continue;
            else {
               flag=1;
               break;
            }
        }
        if(flag==0) {
            print(iteration,values_new,n);  // print final values of unknowns  and return
            cout << "Return with accurate answer with niters = " << iteration << "\n";
            return ;
        }
        flag=0; //resetting the flag

        print(iteration,values_new,n); //To print intermediate roots

        //copy new values of unknowns to old value array
        for(k=0;k<n;k++)
             values_old[k]=values_new[k];
    } //end of iteration loop
    print(iteration,values_new,n) ;
    cout << "Tot number of iterations: " << iteration << endl;

} //end of findValues()


void generateMatrix(vector<vector<float>> &a, int n) {
    srand(time(0)); // Seed for random number generation
    for (int i = 0; i < n; i++) {
        float rowSum = 0; // Sum of off-diagonal elements in the row

        for (int j = 0; j < n; j++) {
            if (i != j) {
                // Generate random off-diagonal elements in the range [-10, 10]
                a[i][j] = static_cast<float>(rand() % 21 - 10);
                rowSum += fabs(a[i][j]);
            }
        }

        // Set diagonal element to ensure diagonal dominance
        a[i][i] = rowSum + static_cast<float>(rand() % 10 + 1); // Slightly larger than rowSum

        // Generate random constant for the last column
        a[i][n] = static_cast<float>(rand() % 100 + 1); // Random constant in the range [1, 100]
    }
}


int main()
{
    int i,j,k,x,y,maxIterations,n;
    float ratio;
    cout<<"Enter no of Unknowns\n";
    cin>>n;
    cout<<"Enter no. of iterations\n";
    cin>>maxIterations;
    vector<vector<float>> a(500, vector<float>(501));
    float values[500];

    generateMatrix(a, n);
    cout<<"Generated the Augmented Matrix\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= n; j++) {
            cout << a[i][j] << " ";
        }
        cout << endl;
    }

   if(!isMethodApplicable(a,n))
   {
        cout<<"\nGauss Seidel Method can't be applied";
        return 0;
   }
   cout<<"\nGauss Seidel Method is applicable\n\n";
   for(int i=0;i<n;i++)
     values[i]=0;
   findValues(a,maxIterations,values,n);
   return 0;
}



