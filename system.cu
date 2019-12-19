#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include"mt.h"

__device__ __constant__ int d_Np;
__device__ __constant__ double d_L;
#define NUM_BLOCK 32
#define NUM_THREAD 32
#define PI 3.1415926535897932384626433

//structure---------------------------------------------------------------------
typedef struct {
	double L;
	double phi;
	double *x;
	double *y;
	int *active;
	int t;
	int Np;
}host_configuration_structure;

typedef struct {
	//Np and L are provided as __device__ __constant__
	double *x;
	double *y;
	int *active;
	int t;
}device_configuration_structure;

typedef struct {
	int *cell_list;
}device_list_structure;

//host functions----------------------------------------------------------------
void init_configuration(double *h_x, double *h_y, double h_L, int h_Np) {
	int i;

	for(i = 0; i < h_Np; i += 1) {
		h_x[i] = h_L * genrand_real2();
		h_y[i] = h_L * genrand_real2();
	}
}

double pbc_distance(double x1, double x2, double L) {
	double dx = x1 - x2;
	double l = 0.5 * L;
	if(dx > l) {
		return dx - L;
	} else if(dx < -l) {
		return L + dx;
	} else {
		return dx;
	}
}

void h_check_active(double *h_x, double *h_y, double h_L, int h_Np, int *h_active) {
	int i, j;
	double dx, dy, dr_square;
	double diameter_square = 1.0;

	for(i = 0; i < h_Np; i += 1) {
		h_active[i] = 0;
	}
	for(i = 0; i < h_Np; i += 1) {
		for(j = 0; j < i; j += 1) {
			dx = h_x[i] - h_x[j];
			if(dx > 0.5 * h_L) {
				dx -= h_L;
			} else if(dx < -0.5 * h_L) {
				dx += h_L;
			}
			dy = h_y[i] - h_y[j];
			if(dy > 0.5 * h_L) {
				dy -= h_L;
			} else if(dy < -0.5 * h_L) {
				dy += h_L;
			}

			dr_square = dx * dx + dy * dy;
			if(dr_square < diameter_square) {
				h_active[i] = 1;
				h_active[j] = 1;
			}
		}
	}
}

void h_DBG(int *A, int *B, int dim) {
	int i;
	double res = 0.0;
	for(i = 0; i < dim; i += 1) {
		res += (A[i] - B[i]) * (A[i] - B[i]);
	}
	printf("res %f\n", res);
}

//device functions--------------------------------------------------------------
__global__ void d_check_active(double *d_x, double *d_y, int *d_active) {
	int i_global, j;
	int Np = d_Np;
	double l = 0.5 * d_L;
	double dx, dy ,dr_square;
	double diameter_square = 1.0;

	i_global = blockDim.x * blockIdx.x + threadIdx.x;

	if (i_global < Np) {
		d_active[i_global] = 1;
		for(j = 0; j < Np; j += 1) {
			dx = d_x[i_global] - d_x[j];
			if(dx > l) {
				dx -= d_L;
			} else if(dx < -l) {
				dx += d_L;
			}
			dy = d_y[i_global] - d_y[j];
			if(dy > l) {
				dy -= d_L;
			} else if(dy < -l) {
				dy += d_L;
			}
			dr_square = dx * dx + dy * dy;

			if(dr_square < diameter_square) {
				d_active[i_global] = 1;
				break;
			}
		}
	}
}

//------------------------------------------------------------------------------
int main(void) {
	int i;
	//host_configuration_structure h_conf;
	double *h_x;
	double *h_y;
	double h_L;
	int *h_active;
	int *h_check_result;
	int h_Np;
	//device_configuration_structure d_conf;
	double *d_x;
	double *d_y;
	int *d_active;
	//initialize
	init_genrand(19970303);
	//--set variable
	h_Np = 280;
	h_L = 25.0;
	h_phi = PI * 0.25 * h_Np / (h_L * h_L);
	cudaMemcpyToSymbol(d_Np, &h_Np, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_L, &h_L, sizeof(double), 0, cudaMemcpyHostToDevice);
	//--allocate memory
	cudaHostAlloc((void **)&h_x, h_Np * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_y, h_Np * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_active, h_Np * sizeof(int), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_check_result, h_Np * sizeof(int), cudaHostAllocMapped);

	cudaMalloc((void **)&d_x, h_Np * sizeof(double));
	cudaMalloc((void **)&d_y, h_Np * sizeof(double));
	cudaMalloc((void **)&d_active, h_Np * sizeof(int));

	//--place particles
	init_configuration(h_x, h_y, h_L, h_Np);
	cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	//--make first acriveness list
	h_check_active(h_x, h_y, h_L, h_Np, h_active);

	d_check_active<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_active);
	cudaDeviceSynchronize();
	cudaMemcpy(h_check_result, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
	printf("\n");

	for(i = 0; i < h_Np; i += 1) {
		printf("(%d,%d) ", h_active[i], h_check_result[i]);
	}
	printf("\n");
	h_DBG(h_active, h_check_result, h_Np);

	//move particles

	//finalize
	//--free memory
       	cudaFreeHost(h_x);
	cudaFreeHost(h_y);
	cudaFreeHost(h_active);
	cudaFree(h_check_result);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_active);
	return 0;
}
