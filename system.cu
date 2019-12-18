#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include"mt.h"

__device__ __constant__ int d_Np;
__device__ __constant__ double d_L;

typedef struct {
	double L;
	double *x;
	double *y;
	double a;
	int t;
	int Np;
}h_config;

typedef struct {
}d_list;

int main(void) {
	return 0;
}
