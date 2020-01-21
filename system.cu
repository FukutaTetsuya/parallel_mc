/*
 * Cell(i, j) = cell[i + j * n]
 */
#include"functions.cuh"

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

//host functions----------------------------------------------------------------
void h_check_active_with_list(double *h_x, double *h_y, double h_L, int h_Np, int *h_active, int *h_cell_list, int cell_per_axis, int N_per_cell) {
	int i, j;
	int x_c, y_c;
	int cell_id, N_in_cell;
	int pair_id;
	double dx, dy, dr_square;
	double diameter_square = 1.0;

	for(i = 0; i < h_Np; i += 1) {
		x_c = (int)(h_x[i] * (double)cell_per_axis / h_L);
		y_c = (int)(h_y[i] * (double)cell_per_axis / h_L);
		cell_id = x_c + y_c * cell_per_axis;
		N_in_cell = h_cell_list[cell_id * N_per_cell];
		for(j = 1; j <= N_in_cell; j += 1) {
			pair_id = h_cell_list[cell_id * N_per_cell + j];
			if(i == pair_id) {continue;}
			dx = h_x[i] - h_x[pair_id];
			if(dx < -0.5 * h_L) {
				dx += h_L;
			} else if(dx > 0.5 * h_L) {
				dx -= h_L;
			}
			dy = h_y[i] - h_y[pair_id];
			if(dy < -0.5 * h_L) {
				dy += h_L;
			} else if(dy > 0.5 * h_L) {
				dy -= h_L;
			}
			dr_square = dx * dx + dy * dy;
			if(diameter_square > dr_square) {
				h_active[i] = 1;
			}
		}
	}
}


int h_make_cell_list(double *h_x, double *h_y, double h_L, int h_Np, int *h_cell_list, int cell_per_axis, int N_per_cell) {
	//I dont know which is better modulo (%)calculation and if(){}elseif(){}else{}
	int i, j, k;
	int x_cell, y_cell;
	int cell_id;
	int cell_list_size = cell_per_axis * cell_per_axis * N_per_cell;
	int contained_num;
	//init cell list
	for(i = 0; i < cell_list_size; i += 1) {
		h_cell_list[i] = 0;
	}
	//make cell list
	for(i = 0; i < h_Np; i += 1) {
		x_cell = (int)(h_x[i] * (double)cell_per_axis / h_L);
		y_cell = (int)(h_y[i] * (double)cell_per_axis / h_L);
		for(j = x_cell - 1; j <= x_cell + 1; j += 1) {
			for(k = y_cell - 1; k <= y_cell + 1; k += 1) {
				cell_id = ((j + cell_per_axis) % cell_per_axis) + ((k + cell_per_axis) % cell_per_axis) * cell_per_axis;
				h_cell_list[cell_id * N_per_cell] += 1;
				contained_num = h_cell_list[cell_id * N_per_cell];
				if(contained_num >= N_per_cell) {
					printf("too many particles in a cell\n");
					return 1;
				}
				h_cell_list[cell_id * N_per_cell + contained_num] = i;
			}
		}
	}

	for(i = 0; i < 10; i += 1) {
	printf("h_cell_list[%d]:%d\n", i * N_per_cell, h_cell_list[i * N_per_cell]);
	}
	return 0;
}
//device functions--------------------------------------------------------------

__global__ void d_check_active_with_list(double *d_x, double *d_y, int *d_active, int *d_cell_list, int cell_per_axis, int N_per_cell) {
	//d_L and d_Np are already declared as __global__ const
	int i, j;
	int x_c, y_c;
	int cell_id, N_in_cell;
	int pair_id;
	int i_global;
	double dx, dy, dr_square;
	double diameter_square = 1.0;
	i_global = blockDim.x * blockIdx.x + threadIdx.x;

	for(i = i_global; i < d_Np; i += NUM_BLOCK * NUM_THREAD) {
		d_active[i] = 0;
		x_c = (int)(d_x[i] * (double)cell_per_axis / d_L);
		y_c = (int)(d_y[i] * (double)cell_per_axis / d_L);
		cell_id = x_c + y_c * cell_per_axis;
		N_in_cell = d_cell_list[cell_id * N_per_cell];	
		for(j = 1; j <= N_in_cell; j += 1) {
			pair_id = d_cell_list[cell_id * N_per_cell + j];
			if(i== pair_id) {continue;}
			dx = d_x[i] - d_x[pair_id];
			dy = d_y[i] - d_y[pair_id];
			if(dx < -0.5 * d_L) {
				dx += d_L;
			} else if(dx > 0.5 * d_L) {
				dx -= d_L;
			}
			if(dy < -0.5 * d_L) {
				dy += d_L;
			} else if(dy > 0.5 * d_L) {
				dy -= d_L;
			}
			dr_square = dx * dx + dy * dy;
			if(diameter_square > dr_square) {
				d_active[i] = 1;
			}
		}
	}
}

__global__ void d_make_belonging_cell_list(double *d_x, double *d_y, int *d_cell_list, int *d_belonging_cell, int cell_per_axis, int N_per_cell) {
	int i;
	int i_global;
	int x_cell, y_cell;
	int cell_id;

	i_global = blockDim.x * blockIdx.x + threadIdx.x;
	for(i = i_global; i < d_Np; i += NUM_BLOCK * NUM_THREAD) {
		x_cell = (int)(d_x[i_global] * (double)cell_per_axis / d_L);
		y_cell = (int)(d_y[i_global] * (double)cell_per_axis / d_L);
		cell_id = x_cell + y_cell * cell_per_axis;
		d_belonging_cell[i_global] = cell_id;
	}
}
__global__ void d_make_cell_list(double *d_x, double *d_y, int *d_cell_list, int *d_belonging_cell, int cell_per_axis, int N_per_cell) {
	int i, j, k, l;
	int i_global;
	int x_cell, y_cell;
	int cell_id;

	i_global = blockDim.x * blockIdx.x + threadIdx.x;

	for(i = i_global; i < cell_per_axis * cell_per_axis; i += NUM_BLOCK * NUM_THREAD) {
		x_cell = i % cell_per_axis;
		y_cell = i / cell_per_axis;
		d_cell_list[i * N_per_cell] = 0;
		for(j = x_cell - 1; j <= x_cell + 1; j += 1) {
			for(k = y_cell - 1; k <= y_cell + 1; k += 1) {
				cell_id = ((j + cell_per_axis) % cell_per_axis) + ((k + cell_per_axis) % cell_per_axis) * cell_per_axis;
				for(l = 0; l < d_Np; l += 1) {
					if(d_belonging_cell[l] == cell_id) {
						d_cell_list[i * N_per_cell] += 1;
						d_cell_list[i * N_per_cell +  d_cell_list[i * N_per_cell] ] = l;
					}
				}
			}
		}
	}
	__syncthreads();
	if(i_global == 0) {
		for(i = 0; i < 10; i += 1) {
		printf("d_cell_list[%d]:%d\n", i * N_per_cell, d_cell_list[i * N_per_cell]);
		}
	}
}


//------------------------------------------------------------------------------
int main(void) {
	//variables in host
	printf("variables in host\n");
	clock_t start, end;
	time_t whole_start, whole_end;
	int cell_per_axis;
	int N_per_cell;
	int t, t_max = 200;
	int check_renew_list;
	double *h_x;
	double *h_y;
	double h_L;
	int *h_active;
	int *h_check_result;
	int h_Np;
	int h_N_active;
	int *h_cell_list;
	int *h_active_DBG;
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
	h_Np = 9900;
	h_L = 140.0;
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
	cudaHostAlloc((void **)&h_check_result, h_Np * sizeof(int), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_cell_list, cell_per_axis * cell_per_axis * N_per_cell * sizeof(int), cudaHostAllocMapped);
	h_active_DBG = (int *)calloc(h_Np, sizeof(int));
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
	printf("----made in host\n");
	start = clock();
	h_check_active(h_x, h_y, h_L, h_Np, h_active);
	end = clock();
	printf("------%d\n", (int)(end - start));

	printf("----made in host with cell list\n");
	start = clock();
	h_make_cell_list(h_x, h_y, h_L, h_Np, h_cell_list, cell_per_axis, N_per_cell);
	h_check_active_with_list(h_x, h_y, h_L, h_Np, h_active_DBG, h_cell_list, cell_per_axis, N_per_cell);
	end = clock();
	printf("------%d, ", (int)(end - start));
	h_DBG(h_active, h_active_DBG, h_Np);	

	printf("----made in device global\n");
	start = clock();
	cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	d_check_active<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_active);
	cudaDeviceSynchronize();
	cudaMemcpy(h_check_result, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
	end = clock();
	printf("------%d, ", (int)(end - start));
	h_DBG(h_active, h_check_result, h_Np);

	/*printf("----made in device global with list, list is made in device\n");
	start = clock();
	cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	d_make_belonging_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_cell_list, d_belonging_cell, cell_per_axis, N_per_cell);
	cudaDeviceSynchronize();
	d_make_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_cell_list, d_belonging_cell, cell_per_axis, N_per_cell);
	cudaDeviceSynchronize();
	d_check_active_with_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_active, d_cell_list, cell_per_axis, N_per_cell);
	cudaDeviceSynchronize();
	cudaMemcpy(h_check_result, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
	end = clock();
	printf("------%d, ", (int)(end - start));
	h_DBG(h_active, h_check_result, h_Np);
	*/

	printf("\ntime loop\n");
	check_renew_list = 5;
	consumed_storage_size = storage_size;
	for(t = 0; t < t_max; t += 1) {
		printf("--t=%d\n", t);
		printf("--count active particles\n");
		//h_N_active = h_reduction_active_array(h_active, h_Np);
		d_N_active = reduction_active_array_on_device(d_active, h_Np);
		//printf("----res_Nactive:%d\n", h_N_active - d_N_active);
		printf("----active frac:%f\n", (double)d_N_active / (double)h_Np);
		printf("----active num:%d\n", d_N_active);

		if(storage_size - consumed_storage_size < d_N_active * 6 || storage_size - consumed_storage_size < h_Np * 0.2) {
			printf("--make kick storage----------------------------------------\n");
			gen_array_kick_on_device(gen_mt, h_kick_storage, storage_size);
			cudaDeviceSynchronize();
			consumed_storage_size = 0;
		}
		printf("--move particles\n");
		h_kick_particles(h_x, h_y, h_active, h_L, h_Np, h_N_active, h_kick_storage, consumed_storage_size);	
		consumed_storage_size += 2 * d_N_active;
		printf("----consumed kick array storage:%d\n", consumed_storage_size);

		if(check_renew_list == 5) {
			printf("--make new cell list\n");
			check_renew_list = 0;
			cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
			d_make_cell_list<<<1, h_Np>>>(d_x, d_y, d_cell_list, d_belonging_cell, cell_per_axis, N_per_cell);
			cudaDeviceSynchronize();
		}
		check_renew_list += 1;

		printf("--check activeness\n");
		//printf("----check activeness on host\n");
		//h_check_active(h_x, h_y, h_L, h_Np, h_active);

		/*printf("----check activeness on device with list\n");
		cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
		d_make_cell_list<<<1, h_Np>>>(d_x, d_y, d_cell_list, d_belonging_cell, cell_per_axis, N_per_cell);
		cudaDeviceSynchronize();
		d_check_active_with_list<<<1, h_Np>>>(d_x, d_y, d_active, d_cell_list, cell_per_axis, N_per_cell);
		cudaDeviceSynchronize();
		cudaMemcpy(h_check_result, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
		printf("----");
		h_DBG(h_active, h_check_result, h_Np);
		*/
		printf("----check activeness on device without list\n");
		cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
		d_check_active<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_active);
		cudaDeviceSynchronize();
		//cudaMemcpy(h_check_result, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_active, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
		//printf("----");
		//h_DBG(h_active, h_check_result, h_Np);
	}



	printf("\nfinalize\n");
	printf("--free memory\n");
	printf("----free memory on host\n");
	cudaFreeHost(h_x);
	cudaFreeHost(h_y);
	cudaFreeHost(h_active);
	cudaFreeHost(h_check_result);
	cudaFreeHost(h_cell_list);
	free(h_active_DBG);
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
