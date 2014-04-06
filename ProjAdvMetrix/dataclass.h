#ifndef BAY_DATA
#define BAY_DATA
#include <stdio.h>
#include <string.h>
#include <gsl\gsl_matrix.h>

//class to read a file and store the data 
class Data {
private:
	int ch; //character read by fgetc()
	FILE * fp; //file pointer

public:
	int size; //number of lines
	int numVar; //number of variables
	gsl_matrix * matrix;

	//Constructor
	Data(char * filename){
		//load file
		fp = fopen(filename,"r");
		char charLine [500] = "0";
		char * pValues;
		double dblValue = 0.00;
		if(fp == NULL){
			printf("File could not be oppened\n");
			delete this;
		}else{
			//count number of entries
			size=0; 
			numVar=1;
			while(!feof(fp)){
				ch = fgetc(fp);
				//count number of variables
				if(size == 0){
					if(ch == ','){numVar++;}
				}
				//count number of lines
				if(ch == '\n'){
					size++;
				}
			}
			size--;//remove header from count
			//create matrix
			matrix = gsl_matrix_alloc (size,numVar);
			rewind(fp); //put the pointer at the begining of the file
			fscanf(fp,"%[^\n]\n",charLine); //skip header
			for(int i=0;i<size;i++){
				fscanf(fp,"%[^\n]\n",charLine); //get the line
				pValues = strtok(charLine,",\t"); //split at first "," or tab
				for(int j=0;j<numVar;j++){
					sscanf(pValues,"%lf",&dblValue); //convert the string to double
					gsl_matrix_set (matrix, i, j, dblValue); //put it in the matrix at position (i,j)
					pValues = strtok(NULL,","); //go to next coma
				}
			}
		
		printf("Data loaded: %d x %d \n",size, numVar);	
		}
	}//end of Constructor

	//METHOD Last10(): print last 10 rows
	void Last5(){
		for (int i = (size-5); i < size; i++){
			printf("[%d] ",i);
			for (int j = 0; j < numVar; j++){
				printf("%.3f\t",gsl_matrix_get(matrix,i,j));
			}
			printf("\n");
		}
	}
};
 
#endif
