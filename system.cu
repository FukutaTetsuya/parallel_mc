#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include"mt.h"

__device__ __constant__ int d_Np;
__device__ __constant__ double d_L;
#define NUM_BLOCK 32
#define NUM_THREAD 32
#define PI 3.14159265358979323846264338327950288412

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
void init_configuration(host_configuration_structure *h_conf) {
	int i;
	double L = h_conf->L;
	int Np = h_conf->Np;

	for(i = 0; i < Np; i += 1) {
		h_conf->x[i] = L * genrand_real2();
		h_conf->y[i] = L * genrand_real2();
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

void h_check_active(host_configuration_structure *h_conf) {
	int i, j;
	int Np = h_conf->Np;
	double dx, dy, dr_square;
	double diameter_square = 1.0;
	double L = h_conf->L;

	for(i = 0; i < Np; i += 1) {
		h_conf->active[i] = 0;
	}
	for(i = 0; i < Np; i += 1) {
		for(j = 0; j < i; j += 1) {
			dx = pbc_distance(h_conf->x[i], h_conf->x[j], L);
			dy = pbc_distance(h_conf->y[i], h_conf->y[j], L);
			dr_square = dx * dx + dy * dy;
			if(dr_square < diameter_square) {
				h_conf->active[i] = 1;
				h_conf->active[j] = 1;
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
	printf("res %d\n", res);
}

//device functions--------------------------------------------------------------
__global__ void d_check_active(device_configuration_structure *d_conf) {
	int i_global, j;
	int Np = d_Np;
	double l = 0.5 * d_L;
	double dx, dy ,dr_square;
	double diameter_square = 1.0;

	i_global = blockDim.x * blockIdx.x + threadIdx.x;

	if (i_global < Np) {
		d_conf->active[i_global] = 0;
		for(j = 0; j < Np; j += 1) {
			dx = d_conf->x[i_global] - d_conf->x[j];
			if(dx > l) {
				dx -= d_L;
			} else if(dx < -l) {
				dx += d_L;
			}
			dy = d_conf->y[i_global] - d_conf->y[j];
			if(dy > l) {
				dy -= d_L;
			} else if(dy < -l) {
				dy += d_L;
			}
			dr_square = dx * dx + dy * dy;

			if(dr_square < diameter_square) {
				d_conf->active[i_global] = 1;
			}
		}
	}
}

//------------------------------------------------------------------------------
int main(void) {
	int i;
	host_configuration_structure h_conf;
	device_configuration_structure d_conf;
	int h_check_result[280] = {0};
	//initialize
	init_genrand(19970303);
	//--set variable
	h_conf.Np = 280;
	h_conf.L = 25.0;
	h_conf.phi = PI * 0.25 * h_conf.Np / (h_conf.L * h_conf.L);
	cudaMemcpyToSymbol(d_Np, &h_conf.Np, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_L, &h_conf.L, sizeof(double), 0, cudaMemcpyHostToDevice);
	//--allocate memory
	cudaHostAlloc((void **)&h_conf.x, h_conf.Np * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_conf.y, h_conf.Np * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_conf.active, h_conf.Np * sizeof(int), cudaHostAllocMapped);
	cudaMalloc((void **)&d_conf.x, h_conf.Np * sizeof(double));
	cudaMalloc((void **)&d_conf.y, h_conf.Np * sizeof(double));
	cudaMalloc((void **)&d_conf.active, h_conf.Np * sizeof(int));
	//--place particles
	init_configuration(&h_conf);
	cudaMemcpy(d_conf.x, h_conf.x, h_conf.Np * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_conf.y, h_conf.y, h_conf.Np * sizeof(double), cudaMemcpyHostToDevice);
	//--make first acriveness list
	h_check_active(&h_conf);

	d_check_active<<<NUM_BLOCK, NUM_THREAD>>>(&d_conf);
	cudaDeviceSynchronize();
	cudaMemcpy(h_check_result, d_conf.active, h_conf.Np * sizeof(int), cudaMemcpyDeviceToHost);

	h_DBG(h_conf.active, h_check_result, h_conf.Np);

	//move particles

	//finalize
	//--free memory
       	cudaFreeHost(h_conf.x);
	cudaFreeHost(h_conf.y);
	cudaFreeHost(h_conf.active);
	cudaFree(d_conf.x);
	cudaFree(d_conf.y);
	cudaFree(d_conf.active);
	return 0;
}
