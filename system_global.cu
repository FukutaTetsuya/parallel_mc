/*
 * Cell(i, j) = cell[i + j * n]
 */
#include"functions_global.cuh"

/*
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<curand.h>
#include"mt.h"

__device__ __constant__ int d_Np;
__device__ __constant__ double d_L;
#define NUM_BLOCK 5
#define NUM_THREAD 1024
#define PI 3.1415926535897932384626433
*/

int main(void) {	
	//variables in host
	printf("variables in host\n");
	time_t whole_start, whole_end;
	int cell_per_axis;
	int N_per_cell;
	int t, t_max = 5;
	int check_renew_list;
	double *h_x;
	double *h_y;
	double h_L;
	int *h_active;
	int h_Np;
	float *h_kick_storage;
	size_t storage_size = 4000000;
	size_t consumed_storage_size;
	curandGenerator_t gen_mt;

	//variables in device
	printf("variables in device\n");
	double *d_x;
	double *d_y;
	int *d_active;
	int *d_cell_list;
	int *d_belonging_cell;
	int d_N_active;

	printf("\ninitialize\n");
	whole_start = time(NULL);
	printf("--set up random number generators\n");
	printf("----host mt\n");
	init_genrand(19970303);
	//init_genrand((unsigned int)time(NULL));
	printf("----cuRAND host API\n");
	curandCreateGenerator(&gen_mt, CURAND_RNG_PSEUDO_MTGP32);
	//curandSetPseudoRandomGeneratorSeed(gen_mt, (unsigned long)time(NULL));
	curandSetPseudoRandomGeneratorSeed(gen_mt, (unsigned long)19970303);
	curandSetGeneratorOffset(gen_mt, 0ULL);
	curandSetGeneratorOrdering(gen_mt, CURAND_ORDERING_PSEUDO_DEFAULT);

	printf("--set parameters\n");
	h_Np = 38000;
	h_L = 280.0;
	cell_per_axis = (int)(h_L / 11.0) + 1;//renew list every 5 steps
	N_per_cell = (h_Np * 13) / (cell_per_axis * cell_per_axis);
	printf("----cell per axis:%d N_per_cell:%d pack. frac. %.5f\n", cell_per_axis, N_per_cell, (double)h_Np * PI / 4.0 / h_L / h_L);

	cudaMemcpyToSymbol(d_Np, &h_Np, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_L, &h_L, sizeof(double), 0, cudaMemcpyHostToDevice);

	printf("--allocate memory\n");
	printf("----memory on host\n");
	cudaHostAlloc((void **)&h_x, h_Np * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_y, h_Np * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_active, h_Np * sizeof(int), cudaHostAllocMapped);
	h_kick_storage = (float *)calloc(storage_size, sizeof(float));
	if(h_kick_storage == NULL) {printf("memory shortage\n"); return 0;}

	printf("----memory on device\n");
	cudaMalloc((void **)&d_x, h_Np * sizeof(double));
	cudaMalloc((void **)&d_y, h_Np * sizeof(double));
	cudaMalloc((void **)&d_active, h_Np * sizeof(int));
	cudaMalloc((void **)&d_cell_list, cell_per_axis * cell_per_axis * N_per_cell * sizeof(int));
	cudaMalloc((void **)&d_belonging_cell, h_Np * sizeof(int));

	printf("--place particles\n");
	init_configuration(h_x, h_y, h_L, h_Np);
	cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);

	printf("--make first acriveness array\n");
	printf("----made in device global with cell list\n");
	cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	d_make_belonging_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_cell_list, d_belonging_cell, cell_per_axis, N_per_cell);
	cudaDeviceSynchronize();
	d_make_cell_list_from_belonging_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_cell_list, d_belonging_cell, cell_per_axis, N_per_cell);
	cudaDeviceSynchronize();
	d_check_active_with_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_active, d_cell_list, cell_per_axis, N_per_cell);
	cudaDeviceSynchronize();
	cudaMemcpy(h_active, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("\ntime loop\n");
	check_renew_list = 5;
	consumed_storage_size = storage_size;
	for(t = 0; t < t_max; t += 1) {
		printf("--t=%d\n", t);
		printf("--count active particles\n");
		//check if reduction works right
		d_N_active = reduction_active_array_on_device(d_active, h_Np);
		printf("----active frac:%f\n", (double)d_N_active / (double)h_Np);
		printf("----active num:%d\n", d_N_active);

		if(storage_size - consumed_storage_size < d_N_active * 6 || storage_size - consumed_storage_size < h_Np * 0.2) {
			printf("--make kick storage\n");
			gen_array_kick_on_device(gen_mt, h_kick_storage, storage_size);
			cudaDeviceSynchronize();
			consumed_storage_size = 0;
		}
		printf("--move particles\n");
		h_kick_particles(h_x, h_y, h_active, h_L, h_Np, d_N_active, h_kick_storage, consumed_storage_size);	
		consumed_storage_size += 2 * d_N_active;
		printf("----consumed kick array storage:%d\n", consumed_storage_size);

		if(check_renew_list == 5) {
			printf("--make new cell list\n");
			check_renew_list = 0;
			cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
			d_make_cell_list_from_belonging_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_cell_list, d_belonging_cell, cell_per_axis, N_per_cell);
			cudaDeviceSynchronize();
		}
		check_renew_list += 1;

		printf("--check activeness\n");
		printf("----check activeness on device with cell list\n");
		cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
		d_check_active_with_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_active, d_cell_list, cell_per_axis, N_per_cell);
		cudaDeviceSynchronize();
		cudaMemcpy(h_active, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
	}



	printf("\nfinalize\n");
	printf("--free memory\n");
	printf("----free memory on host\n");
	cudaFreeHost(h_x);
	cudaFreeHost(h_y);
	cudaFreeHost(h_active);
	free(h_kick_storage);
	printf("----free memory on device\n");
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_active);
	cudaFree(d_cell_list);
	cudaFree(d_belonging_cell);
	printf("--destroy random number generator\n");
	curandDestroyGenerator(gen_mt);

	whole_end = time(NULL);
	printf("%d sec\n", (int)(whole_end - whole_start));
	printf("\nreturn 0;\n");
	return 0;
}
