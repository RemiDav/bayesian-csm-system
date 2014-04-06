#ifndef BAY_MODEL
#define BAY_MODEL
#define DEBUG true
#include <stdio.h>
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include <curand.h>
#include <string.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include "CUDAcBLAS.h"
#define IDX2C(i,j,rows) (((j)*(rows))+(i))
#include <gsl\gsl_matrix.h>
#include <gsl\gsl_rng.h>
#include <gsl\gsl_randist.h>
#include <gsl\gsl_sf_log.h>


#define NUM_J 4 //number of blocks
#define NUM_N 512 //number of threads per block, J*N = number of theta
#define NUM_ENDOG 2
#define NUM_EXOG 4
#define D1 0.5

/*-----Inputs-----*/
#define MEAN_B {1,0,0,-1,2,2,-2,0}
#define VAR_B 1.5
#define MEAN_G -0.5
#define VAR_G 0.5
#define VAR_O 1


gsl_rng * r; //global generator

/*----------------------THETA CLASS & MANAGEMENT----------------*/

class Theta {
public:
	double gamma[NUM_ENDOG*NUM_ENDOG];
	//-double sigma[NUM_ENDOG*NUM_ENDOG]; //to be deleted
	double invLsigma[NUM_ENDOG*NUM_ENDOG];
	double transpInvLomega[NUM_ENDOG*NUM_ENDOG];
	double log_det_omega;
	//-double omega[NUM_ENDOG*NUM_ENDOG]; //to be deleted
	//-double Xsigma[NUM_ENDOG*NUM_ENDOG]; //Matrix used to generate Sigma:   Sigma = alpha X'X
	double beta[NUM_EXOG*NUM_ENDOG];
	double pi[NUM_EXOG*NUM_ENDOG];
	long double logweight;
	int num_endog; //number of endogenous vars
	int num_exog;  //number of exogenous vars
	__host__ __device__ Theta(){
		logweight = 0;
		num_endog = NUM_ENDOG;
		num_exog = NUM_EXOG;
	}
};

class arrayTheta { 
	//stores array of theta parameters pointers. 
	//i.e.: arrayTheta.sigma is an array of pointers to sigma parameters
public:
	double* gamma[NUM_J * NUM_N];
	double* beta[NUM_J * NUM_N];
	double* pi[NUM_J * NUM_N];
	double* omega[NUM_J * NUM_N];
	double** dGamma;
	double** dBeta;
	double** dPi;
	double** dOmega;
	__host__ __device__ arrayTheta(){
		for(int i=0; i<(NUM_N*NUM_J);i++){
			gamma[i] = (double *) malloc(sizeof(double *));
			beta[i] = (double *) malloc(sizeof(double *));
			pi[i] = (double *) malloc(sizeof(double *));
			omega[i] = (double *) malloc(sizeof(double *));
		}
		cudaMalloc((void **)&dGamma,NUM_J * NUM_N *sizeof(double*));
		cudaMalloc((void **)&dBeta,NUM_J * NUM_N *sizeof(double*));
		cudaMalloc((void **)&dPi,NUM_J * NUM_N *sizeof(double*));
		cudaMalloc((void **)&dOmega,NUM_J * NUM_N *sizeof(double*));
	}
	__host__ void copyToDevice(){
		for(int i=0;i<(NUM_J*NUM_N);i++){
			cudaMemcpy(&(dGamma[i]),&(gamma[i]),sizeof(double*),cudaMemcpyHostToDevice);
			cudaMemcpy(&(dBeta[i]),&(beta[i]),sizeof(double*),cudaMemcpyHostToDevice);
			cudaMemcpy(&(dPi[i]),&(pi[i]),sizeof(double*),cudaMemcpyHostToDevice);
			cudaMemcpy(&(dOmega[i]),&(omega[i]),sizeof(double*),cudaMemcpyHostToDevice);
		}
	}
};

/*Copy particles vectors*/
__host__ void copyParticles(Theta * destParticles,Theta * sourceParticles,cudaMemcpyKind MemcpyType){
	for(int i=0;i<(NUM_J*NUM_N);i++){
		cudaMemcpy(&(destParticles[i]),&(sourceParticles[i]), sizeof(Theta), MemcpyType);
	}
}

#include "modelkernells.h"

/*--------Compute Pi = -B.inv(Gamma) ------------*/
void compute_pi(cublasHandle_t handle,Theta * dParticles){
	cublasStatus_t stat;
	printf("Compute Pi = -B.inv(Gamma) .\n");
	//generate array of adresses
	arrayTheta arrTheta;
	for(int i=0; i<(NUM_N*NUM_J);i++){
		arrTheta.gamma[i]=(double *) dParticles[i].gamma;
		arrTheta.pi[i]=(double *) dParticles[i].pi;
	}
	arrTheta.copyToDevice();
	//Copy beta to pi (on the device)
	copy_beta_to_pi<<<NUM_J,NUM_N>>>(dParticles);

	//trsmbatched():  X = a B inv(A) if SIDE_RIGHT;   with a=-1 ; B = B;  A = Gamma (tri);  The solution overwrites B.
	//need to convert to gemm batched if gamma not triangular
	double alpha;
	alpha = -1.0;
	stat = cublasDtrsmBatched(handle, CUBLAS_SIDE_RIGHT, //Gamma is on the right
						CUBLAS_FILL_MODE_UPPER, //Gamma is upper triangular
                        CUBLAS_OP_N, //Gamma does not need transposition
                        CUBLAS_DIAG_UNIT,//Gamma's diag is filled with 1
                        NUM_EXOG,NUM_ENDOG,//Beta is NUM_EXOG x NUM_ENDOG (K x M in Greene 10.6)
						&alpha, //the result is multiplied by -1
						(double **)arrTheta.dGamma, NUM_ENDOG, //Array of A matrices
                        (double **)arrTheta.dPi, NUM_EXOG, //Array of B matrices (and output)
                        NUM_J*NUM_N //Number of particles
						);
	if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("cublasDtrsmBatched failure\n");
    }
}


/*-------------------END THETA CLASS & MANAGEMENT----------------*/


/* CLASS DevMatrix, store 2D double data on device */ 
class devMatrix { //create a column major matrix (ROW-Major order)
public:
	int rows;
	int cols;
	double * dData;
	__host__ devMatrix(){ //create object without allocating, need to call create(i,j) to allocate it

	}

	__host__ devMatrix(int r,int c){ //create object and allocate space
		rows = r;
		cols = c;
		cudaMalloc((void **) &dData, rows * cols * sizeof(double));
		cudaMemset(dData, 0, rows * cols * sizeof(double) ); //set results to 0
	}

	__host__ void create(int r,int c){
		rows = r;
		cols = c;
		cudaMalloc((void **) &dData, rows * cols * sizeof(double));
		cudaMemset(dData, 0, rows * cols * sizeof(double) ); //set results to 0
		printf("Datamatrix memory allocated\n");
	}
	__host__ void set(int i, int j,double value){
		cudaMemcpy(&dData[i*cols+j],&value,sizeof(double),cudaMemcpyHostToDevice);
	}
	__host__ double get(int i, int j){
		double value = 0;
		cudaMemcpy(&value,&dData[i*cols+j],sizeof(double),cudaMemcpyDeviceToHost);
		return value;
	}
};

class Model {
public:
	int num_j;
	int num_n;
	int num_endog;
	int num_exog;
	int max_s; //number of datapoints
	int s; //current t
	double sumW; 
	Theta * hparticlesL;
	Theta * dparticlesL;
	Theta * hparticlesLm1;
	Theta * dparticlesLm1;
	int * dMutationTracking; //vectors of 1 (mutated) or 0 (did not mutate)
	int * hMutationTracking;
	int numberMutations;
	Theta VarianceTheta; //variance in the sample
	gsl_matrix * Yendog,* Xexog; //endogenous and exogenous variables on host
	devMatrix dYendog, dXexog ; //endogenous and exogenous variables on device
	cublasStatus_t stat; //for cuBLAS
	cublasHandle_t handle; //for cuBLAS
	curandState *devStates; //for curand

	
	//constructor
	__host__ Model(int j, int n, int e, int x){
		num_j = j;
		num_n = n;
		num_endog = e;
		num_exog = x;
		s = 0;
		hparticlesL = (Theta *) malloc(sizeof(Theta) * num_j * num_n);
		hparticlesLm1 = (Theta *) malloc(sizeof(Theta) * num_j * num_n);
		cudaMalloc((void **)&dparticlesL, NUM_J * NUM_N * sizeof(Theta));
		cudaMalloc((void **)&dparticlesLm1, NUM_J * NUM_N * sizeof(Theta));
		cudaMemcpy(dparticlesL,hparticlesL, NUM_J * NUM_N * sizeof(Theta), cudaMemcpyHostToDevice); //copy vector of particles pointers to device
		cudaMemcpy(dparticlesLm1,dparticlesL, NUM_J * NUM_N * sizeof(Theta), cudaMemcpyDeviceToDevice); //copy vector of pointers to particles
		//array to track if a mutation happened
		cudaMalloc((void **)&dMutationTracking, NUM_J * NUM_N * sizeof(int));
		hMutationTracking = (int *) malloc(sizeof(int) * num_j * num_n);

		/* Allocate space for prng states on device */
		cudaMalloc((void **)&devStates, NUM_J * NUM_N * sizeof(curandState));
		printf("Model Loaded\n");

		//load cublas
		stat = cublasCreate(&handle);
		if (stat != CUBLAS_STATUS_SUCCESS) {
			printf ("CUBLAS initialization failed\n");
		}
	}

	/*------------------draw theta--------------------*/
	__host__ void DrawTheta(){
		/*Setup RNG*/
		const gsl_rng_type * T;
		gsl_rng_env_setup();
		T = gsl_rng_default;
		r = gsl_rng_alloc (T);
		double det_inv_omega =0;

		printf("\nDraw Theta particles:\n");
		printf ("generator type: %s\n", gsl_rng_name (r));
		printf ("seed = %lu\n", gsl_rng_default_seed);
		for(int j=0;j<num_j;j++){
			for(int n=0;n<num_n;n++){
				Theta tempTheta;

				/*Beta*/
				double betaPriorMean[] = MEAN_B;
				for(int b_i=0; b_i<tempTheta.num_exog; b_i++){
					for(int b_j=0; b_j<tempTheta.num_endog; b_j++){
						tempTheta.beta[b_j*NUM_EXOG+b_i] = betaPriorMean[b_j*NUM_EXOG+b_i] +  gsl_ran_gaussian(r,VAR_B);
					}
				}
				/*Gamma*/
				for(int b_i=0; b_i<tempTheta.num_endog; b_i++){
					for(int b_j=0; b_j<tempTheta.num_endog; b_j++){
						if(b_i==b_j){
							tempTheta.gamma[b_j*NUM_ENDOG+b_i]=1;
						}else if(b_i>b_j){
							tempTheta.gamma[b_j*NUM_ENDOG+b_i]=0;
						}else{
							tempTheta.gamma[b_j*NUM_ENDOG+b_i] = gsl_ran_gaussian(r,VAR_G) + (MEAN_G);
						}
					}
				}

				/*invLSigma*/
				for(int b_i=0; b_i<tempTheta.num_endog; b_i++){
					for(int b_j=0; b_j<NUM_ENDOG; b_j++){
						if(b_i>b_j){
							tempTheta.invLsigma[b_j*NUM_ENDOG+b_i]= gsl_ran_gaussian(r,1);
						}else if(b_i==b_j){
							tempTheta.invLsigma[b_j*NUM_ENDOG+b_i]=sqrt(gsl_ran_chisq(r,NUM_ENDOG-b_i+1));
						}else{
							tempTheta.invLsigma[b_j*NUM_ENDOG+b_i]=0;
						}
					}
				}

				/*det_omega*/
				//compute det(Omega) = 1/det(invOmega) = 1 / {Prod[diag(Gamma)] * Prod[diag(invLsigma)]}^2
				det_inv_omega=1;
				for(int b_i=0; b_i<NUM_ENDOG;b_i++){
					det_inv_omega = det_inv_omega * tempTheta.gamma[b_i*NUM_ENDOG+b_i] * tempTheta.invLsigma[b_i*NUM_ENDOG+b_i] * tempTheta.invLsigma[b_i*NUM_ENDOG+b_i] * tempTheta.gamma[b_i*NUM_ENDOG+b_i] ;
				}

				tempTheta.log_det_omega = -log(det_inv_omega); //ln(det_omega) = ln(1) - ln(det(invOmega))
				if(DEBUG && ((j * num_n)+n)<2 ){printf("log_det_omega[%d]:%.4f\n",(j * num_n)+n,tempTheta.log_det_omega);}


				//copy particle to device
				hparticlesLm1[(j * num_n)+n] = tempTheta;
				cudaMemcpy(&(dparticlesLm1[(j * num_n)+n]),&(hparticlesLm1[(j * num_n)+n]), sizeof(Theta), cudaMemcpyHostToDevice);


				//Compute transpInvLomega =  Gamma . invLsigma'
				// invLsigma is side_right, transposed and lower
				//C= a B op ( A ) 
				/*cublasStatus_t cublasDtrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           double                *C, int ldc)*/
				double alpha = 1;
				//double beta = 0;
				stat = cublasDtrmm(handle,
					CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER, 
					CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
					NUM_ENDOG,NUM_ENDOG,
					&alpha,
					(double *) &(dparticlesLm1[(j * num_n)+n].gamma),NUM_ENDOG,
					(double *) &(dparticlesLm1[(j * num_n)+n].invLsigma),NUM_ENDOG,
					(double *) &(dparticlesLm1[(j * num_n)+n].transpInvLomega),NUM_ENDOG //saves result in dparticlesLm1[(j * num_n)+n].transpInvLomega
				);
				
				
				cudaMemcpy(&(hparticlesLm1[(j * num_n)+n]),&(dparticlesLm1[(j * num_n)+n]), sizeof(Theta), cudaMemcpyDeviceToHost);
			}
		}//end for i,j
		
		gsl_rng_free (r);
		printf("First 2 particles:\n");
		for(int i=0; i<2; i++){
			printf("[%d] beta (%d x %d):\n",i,hparticlesLm1[0].num_exog,hparticlesLm1[0].num_endog);
			for(int b_i=0; b_i<hparticlesLm1[0].num_exog; b_i++){
				for(int b_j=0; b_j<hparticlesLm1[0].num_endog; b_j++){
					printf("%.3f\t ",hparticlesLm1[i].beta[IDX2C(b_i,b_j,NUM_EXOG)]);
				}
			printf("\n");
			}
			printf("\n");
			printf("gamma:");
			for(int b_i=0; b_i<hparticlesLm1[0].num_endog; b_i++){
				for(int b_j=0; b_j<hparticlesLm1[0].num_endog; b_j++){
					printf("%.3f ",hparticlesLm1[i].gamma[IDX2C(b_i,b_j,NUM_ENDOG)]);
				}
			printf("\t");
			}
			printf("\n");
		}

		/*--------Compute Pi = -B.inv(Gamma)   AND    Omega = inv(Gamma)'.Sigma.inv(Gamma)  --------*/
		compute_pi(handle,dparticlesLm1);

		/*--------Compute Omega ------------*/
		copyParticles(hparticlesLm1,dparticlesLm1,cudaMemcpyDeviceToHost);

		//Print Pi for 2 first particles
		for(int i=0; i<2; i++){
			printf("[%d] Pi (%d x %d):\n",i,hparticlesLm1[0].num_exog,hparticlesLm1[0].num_endog);
			for(int b_i=0; b_i<hparticlesLm1[0].num_exog; b_i++){
				for(int b_j=0; b_j<hparticlesLm1[0].num_endog; b_j++){
					printf("%.3f\t ",hparticlesLm1[i].pi[IDX2C(b_i,b_j,NUM_EXOG)]);
				}
			printf("\n");
			}
			printf("\n");
		}
	}
	/*------------------Load variables--------------------*/
	//variables are stored on the device in ROW-Major order
	__host__ void LoadData(gsl_matrix * SourceData){
		printf("\nLoad data series:\n");
		double tempValue;
		max_s = SourceData[0].size1;
		//allocate memory variables on host
		Yendog = gsl_matrix_alloc(max_s,NUM_ENDOG);
		Xexog = gsl_matrix_alloc(max_s,NUM_EXOG);
		
		//create matrices for variables on device
		dYendog.create(max_s,NUM_ENDOG);
		dXexog.create(max_s,NUM_EXOG);
		

		gsl_vector * tempVector = gsl_vector_alloc(max_s);
		
		for(int j=0;j<NUM_ENDOG;j++){ //load endog var data on host
			gsl_matrix_get_col(tempVector,SourceData,j);
			gsl_matrix_set_col(Yendog,j,tempVector);
		}
		for(int j=NUM_ENDOG;j < (NUM_EXOG+NUM_ENDOG) ;j++){ //load exog var data on host
			gsl_matrix_get_col(tempVector,SourceData,j);
			gsl_matrix_set_col(Xexog,(j-NUM_ENDOG),tempVector);
		}

		for(int i=0; i<max_s;i++){
			for(int j=0;j<NUM_ENDOG;j++){ //load endog var data on device
				tempValue = gsl_matrix_get(Yendog,i,j);
				dYendog.set(i,j,tempValue);
			}
			for(int j=0;j<NUM_EXOG;j++){ //load exog var data on device
				tempValue = gsl_matrix_get(Xexog,i,j);
				dXexog.set(i,j,tempValue);
			}
		}
		
		printf("Host: Y1[0] Y2[0]: \t%f\t%f\n",gsl_matrix_get(Yendog,0,0),gsl_matrix_get(Yendog,0,1));
		printf("Dev: Y1[0] Y2[0]: \t%f\t%f\n",dYendog.get(0,0),dYendog.get(0,1));
		printf("Host: X1-4[0]: \t%f\t%f\t%f\t%f\n",gsl_matrix_get(Xexog,0,0),gsl_matrix_get(Xexog,0,1),gsl_matrix_get(Xexog,0,2),gsl_matrix_get(Xexog,0,3));
		printf("Dev: X1-4[0]: \t%f\t%f\t%f\t%f\n",dXexog.get(0,0),dXexog.get(0,1),dXexog.get(0,2),dXexog.get(0,3));
	}

	/*------------------update logweight--------------*/
	__host__ void UpdateLogWeight(){
		kUpdateLogWeight<<<NUM_J,NUM_N>>>(s,dparticlesLm1,dYendog.dData,dXexog.dData);
		cudaDeviceSynchronize();
		copyParticles(hparticlesLm1,dparticlesLm1,cudaMemcpyDeviceToHost);
	}

	/*---------------Reset logweight (Lm1)----------------*/
	__host__ void resetWeight(){
		for(int i=0;i<(NUM_J*NUM_N);i++){
			hparticlesLm1[i].logweight=0;
		}
		copyParticles(dparticlesLm1,hparticlesLm1,cudaMemcpyHostToDevice);
	}

	/*---------------Compute ESS (Lm1)----------------*/
	__host__ double computeESS(){
		double ess = 0;

		//Get max logweight
		long double maxLogW = hparticlesLm1[0].logweight;
		for(int i = 0; i < (NUM_J * NUM_N); i+=1) {
			if(maxLogW<hparticlesLm1[i].logweight){
				maxLogW=hparticlesLm1[i].logweight;
			}
		}
		printf("Maximum LogWeight: %.4f\n",maxLogW);

		//compute sum(w) and sum(w^2)
		sumW=0;
		double sumSqW=0;
		for(int i = 0; i < (NUM_J * NUM_N); i+=1) {
			hparticlesLm1[i].logweight=hparticlesLm1[i].logweight-maxLogW;
			if(hparticlesLm1[i].logweight == hparticlesLm1[i].logweight){ //add only if finite
				sumW += exp(hparticlesLm1[i].logweight);
				sumSqW += exp(hparticlesLm1[i].logweight) * exp(hparticlesLm1[i].logweight);
			}
			//printf("w[%d]=%f\n",i,hparticlesLm1[i].logweight);
		}
		ess = ((sumW * sumW) / sumSqW) / (NUM_J * NUM_N);
		return ess;
	}

	/*--------------Resample----------------*/
	__host__ void resample(){
		long double lowerBound[(NUM_J*NUM_N+1)];
		lowerBound[0]=0;
		for(int i = 0; i < (NUM_J * NUM_N); i++) {
			lowerBound[i+1] = lowerBound[i] + exp(hparticlesLm1[i].logweight) / sumW;
			//if(i<10){printf("interval[%d]: %f->%f for w=%f\n",i,lowerBound[i],lowerBound[i+1],exp(hparticlesLm1[i].logweight));}
		}

		//draw Thetas
		double random_uniform = 0;
		const gsl_rng_type * T;
		gsl_rng_env_setup();
		T = gsl_rng_default;
		r = gsl_rng_alloc (T);
		int source = 0;
		for(int ii=0; ii < (NUM_J * NUM_N); ii++) { //each i is a different draw resulting in a new hparticlesL[i]
			random_uniform = gsl_ran_flat (r, 0, 1);
			for(int j=0,k=0,source=0; k==0; j++){
				if(random_uniform<=lowerBound[j]){//increase j until random_uniform<=lowerBound[j]
					source = j-1;
					if(ii<40){printf("-From [%d] to [%d], LogWeight: %.5f  \n",source,ii,hparticlesLm1[source].logweight );}
					hparticlesL[ii]=hparticlesLm1[source]; //the particle ThetaL[i] gets the value of the particle ThetaLm1[source]
					k=1;
				}
			}
		}
		//L becomes Lm1
		copyParticles(hparticlesLm1,hparticlesL,cudaMemcpyHostToHost);
		copyParticles(dparticlesLm1,hparticlesLm1,cudaMemcpyHostToDevice);
		gsl_rng_free (r);

	}

	/*--------------Compute Variance Lm1------------------*/
	__host__ void computeVariance(){
		Theta sum;
		Theta sumsq;
		//initialize
		for(int i = 0; i< (NUM_ENDOG * NUM_EXOG);i++){
			sum.beta[i]=0;
			sumsq.beta[i]=0;
		}
		for(int i = 0; i< (NUM_ENDOG*NUM_ENDOG);i++){
			sum.gamma[i]=0;
			sumsq.gamma[i]=0;
			sum.invLsigma[i]=0;
			sumsq.invLsigma[i]=0;
		}
		//sum and sumsq
		for(int i = 0; i < (NUM_J * NUM_N); i++) {
			for(int j = 0; j< (NUM_ENDOG * NUM_EXOG);j++){
				sum.beta[j]+=hparticlesLm1[i].beta[j];
				sumsq.beta[j]+=hparticlesLm1[i].beta[j]*hparticlesLm1[i].beta[j];
			}
			for(int j = 0; j< (NUM_ENDOG*NUM_ENDOG);j++){
				sum.gamma[j]+= hparticlesLm1[i].gamma[j];
				sumsq.gamma[j]+= hparticlesLm1[i].gamma[j] * hparticlesLm1[i].gamma[j];
			}
			/*invLSigma*/
			for(int b_i=0; b_i<hparticlesL[i].num_endog; b_i++){
				for(int b_j=0; b_j<NUM_ENDOG; b_j++){
					if(b_i>b_j){
						sum.invLsigma[b_j*NUM_ENDOG+b_i]+= hparticlesLm1[i].invLsigma[b_j*NUM_ENDOG+b_i];
						sumsq.invLsigma[b_j*NUM_ENDOG+b_i]+= hparticlesLm1[i].invLsigma[b_j*NUM_ENDOG+b_i] * hparticlesLm1[i].invLsigma[b_j*NUM_ENDOG+b_i];
					}else if(b_i==b_j){ //variance of the log(diag)
						sum.invLsigma[b_j*NUM_ENDOG+b_i]+= log(hparticlesLm1[i].invLsigma[b_j*NUM_ENDOG+b_i]);
						sumsq.invLsigma[b_j*NUM_ENDOG+b_i]+= log(hparticlesLm1[i].invLsigma[b_j*NUM_ENDOG+b_i]) * log(hparticlesLm1[i].invLsigma[b_j*NUM_ENDOG+b_i]);
					}
				}
			}
		}
		//get variance
		for(int i = 0; i< (NUM_ENDOG * NUM_EXOG);i++){
			VarianceTheta.beta[i]= sumsq.beta[i] / (num_j * num_n) - (sum.beta[i]*sum.beta[i]) / (num_j *num_j * num_n * num_n) ;
		}
		for(int i = 0; i< (NUM_ENDOG*NUM_ENDOG);i++){
			VarianceTheta.gamma[i]= sumsq.gamma[i] / (num_j * num_n) - sum.gamma[i]*sum.gamma[i] / (num_j *num_j * num_n * num_n) ;
			VarianceTheta.invLsigma[i]= sumsq.invLsigma[i] / (num_j * num_n) - (sum.invLsigma[i]*sum.invLsigma[i]) / (num_j *num_j * num_n * num_n) ;
		}
		printf("\nVariance Beta:\n%.4f\t%.4f\n%.4f\t%.4f\n%.4f\t%.4f\n%.4f\t%.4f\n",
			VarianceTheta.beta[0],VarianceTheta.beta[4],
			VarianceTheta.beta[1],VarianceTheta.beta[5],
			VarianceTheta.beta[2],VarianceTheta.beta[6],
			VarianceTheta.beta[3],VarianceTheta.beta[7]			
		);
		printf("\nVariance Gamma:\n%.4f\t%.4f\n%.4f\t%.4f\n",
			VarianceTheta.gamma[0],VarianceTheta.gamma[2],
			VarianceTheta.gamma[1],VarianceTheta.gamma[3]	
		);
		printf("\nVariance invLsigma (log diag):\n%.4f\t%.4f\n%.4f\t%.4f\n",
			VarianceTheta.invLsigma[0],VarianceTheta.invLsigma[2],
			VarianceTheta.invLsigma[1],VarianceTheta.invLsigma[3]	
		);
		
	}

	
	/*--------------Compute Mean Lm1------------------*/
	__host__ void computeMean(){
		Theta sum;
		Theta mean;
		//initialize
		for(int i = 0; i< (NUM_ENDOG * NUM_EXOG);i++){
			sum.beta[i]=0;
		}
		for(int i = 0; i< (NUM_ENDOG*NUM_ENDOG);i++){
			sum.gamma[i]=0;
		}
		//sum
		for(int i = 0; i < (NUM_J * NUM_N); i++) {
			for(int j = 0; j< (NUM_ENDOG * NUM_EXOG);j++){
				sum.beta[j]+=hparticlesLm1[i].beta[j];
			}
			for(int j = 0; j< (NUM_ENDOG*NUM_ENDOG);j++){
				sum.gamma[j]+= hparticlesLm1[i].gamma[j];
			}
		}
		//get mean
		for(int i = 0; i< (NUM_ENDOG * NUM_EXOG);i++){
			mean.beta[i]= sum.beta[i] / (num_j * num_n);
		}
		for(int i = 0; i< (NUM_ENDOG*NUM_ENDOG);i++){
			mean.gamma[i]= sum.gamma[i] / (num_j * num_n);
		}
		printf("\nMean Beta:\n%.4f\t%.4f\n%.4f\t%.4f\n%.4f\t%.4f\n%.4f\t%.4f\n",
			mean.beta[0],mean.beta[4],
			mean.beta[1],mean.beta[5],
			mean.beta[2],mean.beta[6],
			mean.beta[3],mean.beta[7]			
		);
		printf("\nMean Gamma:\n%.4f\t%.4f\n%.4f\t%.4f\n",
			mean.gamma[0],mean.gamma[2],
			mean.gamma[1],mean.gamma[3]	
		);
		
	}

	/*--------------Mutate------------------*/
	__host__ void mutate(){
		const gsl_rng_type * T;
		gsl_rng_env_setup();
		T = gsl_rng_default;
		r = gsl_rng_alloc (T);
		double red_coef = 6.0;
		double det_inv_omega = 0;

		
		for(int i = 0; i < (NUM_J * NUM_N); i++) {
			//Step in beta space
			for(int j = 0; j< (NUM_ENDOG * NUM_EXOG);j++){
				hparticlesL[i].beta[j]=hparticlesLm1[i].beta[j] + gsl_ran_gaussian(r,sqrt(VarianceTheta.beta[j])/red_coef);
			}
			//Step in gamma space
			for(int b_i=0; b_i<NUM_ENDOG; b_i++){
				for(int b_j=0; b_j<NUM_ENDOG; b_j++){
					if(b_i<b_j){
						hparticlesL[i].gamma[b_j*NUM_ENDOG+b_i] = hparticlesLm1[i].gamma[b_j*NUM_ENDOG+b_i] + gsl_ran_gaussian(r,sqrt(VarianceTheta.gamma[b_j*NUM_ENDOG+b_i])*1.5/red_coef);
					}
				}
			}

			//Step in invLSigma space
			/*invLSigma*/
			for(int b_i=0; b_i<hparticlesL[i].num_endog; b_i++){
				for(int b_j=0; b_j<NUM_ENDOG; b_j++){
					if(b_i>b_j){
						hparticlesL[i].invLsigma[b_j*NUM_ENDOG+b_i]= hparticlesLm1[i].invLsigma[b_j*NUM_ENDOG+b_i] + gsl_ran_gaussian(r,sqrt(VarianceTheta.invLsigma[b_j*NUM_ENDOG+b_i])/red_coef);
					}else if(b_i==b_j){
						hparticlesL[i].invLsigma[b_j*NUM_ENDOG+b_i]= exp( log(hparticlesLm1[i].invLsigma[b_j*NUM_ENDOG+b_i]) + gsl_ran_gaussian(r,sqrt(VarianceTheta.invLsigma[b_j*NUM_ENDOG+b_i])/red_coef));
					}
				}
			}
			/*det_omega*/
			//compute det(Omega) = 1/det(invOmega) = 1 / {Prod[diag(Gamma)] * Prod[diag(invLsigma)]}^2
			det_inv_omega=0;
			for(int b_i=0; b_i<NUM_ENDOG;b_i++){
				det_inv_omega = hparticlesL[i].gamma[b_i*NUM_ENDOG+b_i] * hparticlesL[i].invLsigma[b_i*NUM_ENDOG+b_i] * hparticlesL[i].invLsigma[b_i*NUM_ENDOG+b_i] * hparticlesL[i].gamma[b_i*NUM_ENDOG+b_i] ;
			}
			hparticlesL[i].log_det_omega = -log(det_inv_omega); //ln(det_omega) = ln(1) - ln(det(invOmega))


		}
		
		
		//send mutated particle for Metropolis-Hastings
		copyParticles(dparticlesL,hparticlesL,cudaMemcpyHostToDevice);
		copyParticles(dparticlesLm1,hparticlesLm1,cudaMemcpyHostToDevice);

		compute_pi(handle,dparticlesL);

		//Compute transpInvLomega =  Gamma . invLsigma'
		for(int i = 0; i < (NUM_J * NUM_N); i++) {
			double alpha = 1;
			double beta = 0;
			stat = cublasDtrmm(handle,
				CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER, 
				CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
				NUM_ENDOG,NUM_ENDOG,
				&alpha,
				(double *) &(dparticlesL[i].gamma),NUM_ENDOG,
				(double *) &(dparticlesL[i].invLsigma),NUM_ENDOG,
				(double *) &(dparticlesL[i].transpInvLomega),NUM_ENDOG //saves result in dparticlesL[(j * num_n)+n].transpInvLomega
			);
		}

		//generate random numbers for mutation:
		time_t curtime = time(NULL);
		setup_rand_kernel<<<NUM_J,NUM_N>>>(devStates,curtime + 1234);

		kHastingsRatio<<<NUM_J,NUM_N>>>(devStates,s,dparticlesLm1,dparticlesL,dYendog.dData,dXexog.dData,dMutationTracking);
		cudaDeviceSynchronize();
		//compute number or mutations
		cudaMemcpy(hMutationTracking,dMutationTracking, NUM_J * NUM_N * sizeof(int), cudaMemcpyDeviceToHost);
		int sum_mut = 0;
		for(int i = 0; i < (NUM_J * NUM_N); i++) {
			sum_mut=sum_mut+hMutationTracking[i];
		}
		numberMutations=sum_mut;
		printf("Number of mutations: %d/%d\n",sum_mut,(NUM_J * NUM_N));
		
		//L becomes Lm1
		copyParticles(dparticlesLm1,dparticlesL,cudaMemcpyDeviceToDevice);
		copyParticles(hparticlesLm1,dparticlesLm1,cudaMemcpyDeviceToHost);
		

		gsl_rng_free (r);
	}

	
};

#endif