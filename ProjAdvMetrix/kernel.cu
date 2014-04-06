
#include "cuda_runtime.h"
#include "formattext.h"
#pragma comment(lib,"cublas.lib")
#include <cublas_v2.h>
#include "device_launch_parameters.h"
#include "dataclass.h"
#include "modelclass.h"

#include <stdio.h>


int main(){
	printf_blue("CSM Algorithm - system of equations\n");

	//load cuBLAS
    cublasStatus_t stat;
    cublasHandle_t cublashandle;
	stat = cublasCreate(&cublashandle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("ERROR: CUBLAS initialization failed\n");
    }else{
		printf_green ("CUBLAS initialized\n");
	}

	//Load Data
   Data MyData("new-output.txt"); //load data, with header, separated by "," or tab with endogenous variables first
   //print last 10 rows of data matrix
   MyData.Last5();
   Model MyModel(NUM_J,NUM_N,NUM_ENDOG,NUM_EXOG);
   MyModel.DrawTheta();
   MyModel.LoadData(MyData.matrix);

   /*-------Start CSM Loop-------*/
   int L = 150;
   int s = 0;
   double ess=0;
   MyModel.s = 50;
   for(int l = 1; l<=L && MyModel.s<(MyData.size-1); l++){
	   /*---C Phase---*/
		printf_blue("\nC Phase:\n");
		printf("(l=%d)\n",l);
		MyModel.resetWeight();
		printf("Starting at s=t[%d]+1: %d \n",l,(s+1));
		do{
			//update logweight with next datapoint
			MyModel.UpdateLogWeight();
			MyModel.s++;
			//compute ESS / JN
			ess = MyModel.computeESS();
			printf("ESS/(J.N)= %.4f @ s=%d\n",ess,MyModel.s);
		}while(ess >= D1 && MyModel.s<MyData.size);

		printf("first 20 particles:\n");
		for(int i=0;i<10;i++){
			printf("Logweight[%d]:%.4f\t\tLogweight[%d]:%.4f\n",(2*i),MyModel.hparticlesLm1[(2*i)].logweight,(2*i)+1,MyModel.hparticlesLm1[(2*i)+1].logweight);
		}
		MyModel.computeMean();
		/*---END of C Phase---*/
		
		/*---S Phase---*/
		printf_blue("\nS Phase:\n");
		MyModel.resample();

		/*---M Phase---*/
		printf_blue("\nM Phase:\n");
		//--------------------HOW MANY MUTATIONS-------------------------------------------------------
		MyModel.computeMean();
		MyModel.computeVariance();
		do{
			for(int i=0;i<4;i++){
				printf("L=%d\ts=%d\ti=%d\n",l,MyModel.s,i);
				MyModel.mutate();
			}
			
			MyModel.computeMean();
			MyModel.computeVariance();
		}while(MyModel.numberMutations>=(MyModel.num_j*MyModel.num_n/20));

		if(MyModel.VarianceTheta.beta[0]<0.000001){break;}
   }

   printf_blue("\nEnd of CSM loops\n");
   printf("Last s:%d\n",MyModel.s);
   MyModel.computeMean();
   MyModel.computeVariance();
   //end of main
   cublasDestroy(cublashandle);
   printf("\nPress enter to exit the program\n");
   getchar();
   return 0;
}

