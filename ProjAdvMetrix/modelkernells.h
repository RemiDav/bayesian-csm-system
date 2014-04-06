#ifndef BAY_KERNELL
#define BAY_KERNELL

#include "cuda_runtime.h"
#include "curand_kernel.h"
#include <curand.h>

//setup a seed for random number generation on the device
__global__ void setup_rand_kernel(curandState *state, int init_num){
    int id = threadIdx.x + blockIdx.x * NUM_N;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(init_num, id, 0, &state[id]);
}


/*copy beta to pi (on the device)*/
__global__ void copy_beta_to_pi(Theta * dParticles){
	int id = blockIdx.x * NUM_N + threadIdx.x;
	for(int i=0; i<(NUM_EXOG*NUM_ENDOG);i++){
		dParticles[id].pi[i] = dParticles[id].beta[i];
	}
}

/*evaluate log normal(x,sqrt_precision)*/
__device__ double log_centered_normal(double x, double sqrt_precision){
	double result;
	double constant = -0.91893853320467274178032973; // ~Log(1/sqrt[2.Pi])
	result = log(sqrt_precision) + constant - 0.5*x*x*sqrt_precision*sqrt_precision;
	return result;
}

/*compute and returns the loglikelihood for a time t: P(y_t|Theta, X_t) */
__device__ double loglikelihood(int s,int id, Theta * dparticlesLm1, double * Yendog, double * Xexog){
	double v[NUM_ENDOG]; //v[i] = Yendog[i] - x(s,:)* Pi(:,i)
	double u[NUM_ENDOG];
	double sum_square_u=0;
	double y_hat[NUM_ENDOG]; //Y_hat = X . Pi
	double log_likelihood; // Log[P(v|theta,x,y)]

	Theta localTheta = dparticlesLm1[id];
	log_likelihood = 0;
	//v = Yendog - Y_hat
	//Y_hat = Pi' . xt
	for(int i=0; i<NUM_ENDOG;i++){
		v[i]=0;
		y_hat[i]=0;
		for(int j=0; j<NUM_EXOG;j++){
			y_hat[i] = y_hat[i] + Xexog[s*NUM_EXOG+j]*localTheta.pi[i*NUM_EXOG+j];
		}
		v[i] = Yendog[s*NUM_ENDOG+i] - y_hat[i];
	}

	//u = invLsigma . Gamma' . vt = invLomega . vt
	for(int i=0; i<NUM_ENDOG;i++){
		u[i]=0;
		for(int j=0; j<NUM_ENDOG;j++){
			u[i] = u[i] + localTheta.transpInvLomega[i*NUM_ENDOG+j] * v[j];
		}
		sum_square_u = sum_square_u + u[i]*u[i];
	}
	//log_likelihood = - 0.5 Ln(det_omega) - NUM_ENDOG * 0.5 * Ln(2 Pi) - 0.5 sum_square_u
	//0.5 * Ln(2 Pi) = 0.918939
	log_likelihood = -0.5*localTheta.log_det_omega - NUM_ENDOG * 0.918939 - 0.5 * sum_square_u;

	return log_likelihood;
}

/*Compute Log[P(theta)] from the prior*/
__device__ double logprior(int id, Theta * dparticlesLm1){
	double log_p_theta = 0;
	double precision = sqrt(1.0/VAR_B);
	double betaPriorMean[] = MEAN_B;
	//for beta
	for(int b_i=0; b_i<NUM_EXOG; b_i++){
		for(int b_j=0; b_j<NUM_ENDOG; b_j++){
			 log_p_theta += log_centered_normal(dparticlesLm1[id].beta[b_j*NUM_EXOG+b_i]-betaPriorMean[b_j*NUM_EXOG+b_i],precision);
		}
	}
	//for gamma
	precision = sqrt(1.0/VAR_G);
	for(int b_i=0; b_i<NUM_ENDOG; b_i++){
		for(int b_j=0; b_j<NUM_ENDOG; b_j++){
			if(b_i<b_j){
				log_p_theta += log_centered_normal( dparticlesLm1[id].gamma[b_j*NUM_ENDOG+b_i] - (MEAN_G),VAR_G);
			}
		}
	}

	//for omega
	/*--------------------------------------------TO ADD,LOG[P(OMGEA)] EVALUATION--------------------------------------------*/

	return log_p_theta;
}

/*debug kernell*/
__global__ void kDebug(Theta * dparticlesLm1){
	int id = blockIdx.x * NUM_N + threadIdx.x;
	if(id==0){
		printf("Theta[%d,%d]: gamma: %.3f \t %.3f \n %.3f \t %.3f \n", blockIdx.x, threadIdx.x, 
			dparticlesLm1[id].gamma[IDX2C(0,0,NUM_ENDOG)],
			dparticlesLm1[id].gamma[IDX2C(1,0,NUM_ENDOG)],
			dparticlesLm1[id].gamma[IDX2C(0,1,NUM_ENDOG)],
			dparticlesLm1[id].gamma[IDX2C(1,1,NUM_ENDOG)]
			);
	}
}

/*update weights and save them in dLm1*/
__global__ void kUpdateLogWeight(int s, Theta * dparticlesLm1, double * Yendog, double * Xexog){
	int id = blockIdx.x * NUM_N + threadIdx.x;

	double log_likelihood; // Log[P(v|theta,x,y)]
	log_likelihood = loglikelihood(s,id,dparticlesLm1,Yendog,Xexog);

	if(id<2){
		printf("Log Likelihood[%d], s=%d: %.5f\n",id,s,log_likelihood);
	}
	dparticlesLm1[id].logweight = dparticlesLm1[id].logweight + log_likelihood;
}

/*step and evaluate*/
__global__ void kHastingsRatio(curandState * state1,int s,Theta * dparticlesLm1,Theta * dparticlesL, double * Yendog, double * Xexog,int * dMutationTracking){
	int id = blockIdx.x * NUM_N + threadIdx.x;
	/* Copy state to local memory for efficiency */
    curandState localState1 = state1[id];
	//compute Log[ P(Y[0:s]|Theta,X) ] = Sum{ Log[ P(y_t|Theta,X) ] }
	double logPyMut = 0; //with mutated Theta
	double logPyStem = 0; //with stem Theta
	for(int t=0;t<s;t++){ //for each datapoint
		logPyMut += loglikelihood(t,id,dparticlesL,Yendog,Xexog);
		logPyStem += loglikelihood(t,id,dparticlesLm1,Yendog,Xexog);
	}

	//compute log[P(Theta)] from the prior
	double logPThetaStem, logPThetaMut; //evaluation from the prior
	logPThetaMut = logprior(id,dparticlesL);
	logPThetaStem = logprior(id,dparticlesLm1);

	//(proportional) evaluation of log[p(theta|y)] and log[p(theta mutated|y)]
	// log[P(y)] =prop= log[P(Theta)] + Log[P(Y|Theta)]
	double logPStem, logPMut;
	logPMut = logPThetaMut + logPyMut;
	logPStem = logPThetaStem + logPyStem;

	//compute log[Hastings Ratio]
	double logHratio=0;
	logHratio = logPMut - logPStem;
	bool mutate = true;
	double rndDouble;
	if( logHratio < 0){
		rndDouble = curand_uniform(&localState1);
		if(rndDouble>exp(logHratio)){ //keep stem if logPMut < logPStem  AND rndDouble < HRatio
			dparticlesL[id] = dparticlesLm1[id];
			mutate = false;
		}
	}
	if(mutate){
		dMutationTracking[id] = 1;
	}else{
		dMutationTracking[id] = 0;
	}
	if(id<10){
		if(mutate){
			printf("log Hastings Ratio[%d]: %.6f - Mutate\n",id,logHratio);
		}else{
			printf("log Hastings Ratio[%d]: %.6f - Keep Stem\n",id,logHratio);
		}
	}

}

#endif