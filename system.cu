#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include"mt.h"

__device__ __constant__ int d_Np;
__device__ __constant__ double d_L;
#define NUM_BLOCK 32
#define NUM_THREAD 32
#define PI 3.1415926535897932384626433

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
	int i_global;
	int i, j;
	int Np = d_Np;
	double l = 0.5 * d_L;
	double dx, dy ,dr_square;
	double diameter_square = 1.0;

	i_global = blockDim.x * blockIdx.x + threadIdx.x;
	for(i = i_global; i < Np; i += NUM_BLOCK * NUM_THREAD) {
		d_active[i] = 0;
		for(j = 0; j < Np; j += 1) {
			if(j != i) {
				dx = d_x[i] - d_x[j];
				if(dx > l) {
					dx -= d_L;
				} else if(dx < -l) {
					dx += d_L;
				}
				dy = d_y[i] - d_y[j];
				if(dy > l) {
					dy -= d_L;
				} else if(dy < -l) {
					dy += d_L;
				}
				dr_square = dx * dx + dy * dy;

				if(dr_square < diameter_square) {
					d_active[i] = 1;
					break;
				}
			}
		}

	}
}

//------------------------------------------------------------------------------
int main(void) {
	int i;
	clock_t start, end;

	double *h_x;
	double *h_y;
	double h_L;
	int *h_active;
	int *h_check_result;
	int h_Np;

	double *d_x;
	double *d_y;
	int *d_active;

	//initialize
	init_genrand((int)time(NULL));

	//--set variable
	h_Np = 36000;
	h_L = 280.0;
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
	start = clock();
	h_check_active(h_x, h_y, h_L, h_Np, h_active);
	end = clock();
	printf("%d [ms]\n", (int)((end - start)*1000 /CLOCKS_PER_SEC ));
	start = clock();
	d_check_active<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_active);
	cudaDeviceSynchronize();
	cudaMemcpy(h_check_result, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
	end = clock();
	printf("%d [ms]\n", (int)((end - start)*1000 /CLOCKS_PER_SEC ));
	
	//printf("\n");
	//for(i = 0; i < h_Np; i += 1) {
	//	printf("(%d,%d) ", h_active[i], h_check_result[i]);
	//}
	//printf("\n");
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
