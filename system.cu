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

//device functions--------------------------------------------------------------

__global__ void d_check_active_with_cell_list(double *d_x, double *d_y, int *d_active, int *d_cell_list, int cell_per_axis, int N_per_cell) {
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
		x_cell = (int)(d_x[i] * (double)cell_per_axis / d_L);
		y_cell = (int)(d_y[i] * (double)cell_per_axis / d_L);
		cell_id = x_cell + y_cell * cell_per_axis;
		d_belonging_cell[i] = cell_id;
	}
}
__global__ void d_check_active_with_belonging_cell_list(double *d_x, double *d_y, int *d_active, int *d_belonging_cell, int cell_per_axis, int N_per_cell) {
	int i, j, k;
	int i_global;
	int x_cell, y_cell;
	int pair_x_cell, pair_y_cell;
	int surrounding_cell_id[9];
	double x, y;
	double dx, dy;
	double diameter_square = 1.0;
	double half_L = d_L * 0.5;

	i_global = blockDim.x * blockIdx.x + threadIdx.x;
	for(i = i_global; i < d_Np; i += NUM_BLOCK * NUM_THREAD) {
		x = d_x[i];
		y = d_y[i];
		d_active[i] = 0;
		x_cell = (int)(x * (double)cell_per_axis / d_L);
		y_cell = (int)(y * (double)cell_per_axis / d_L);

		//ID of surrounding 9 cells
		for(j = -1; j <= 1; j += 1) {
			pair_x_cell = (x_cell + j + cell_per_axis) % cell_per_axis;
			for(k = -1; k <= 1; k += 1) {
				pair_y_cell = (y_cell + k + cell_per_axis) % cell_per_axis;
				surrounding_cell_id[(j + 1) + (k + 1) * 3] = pair_x_cell + pair_y_cell * cell_per_axis;
			}
		}

		for(j = 0; j < d_Np; j += 1) {
			if(i == j) {continue;}
			for(k = 0; k < 9; k += 1) {
				if(surrounding_cell_id[k] == d_belonging_cell[j]) {
					dx = x - d_x[j];
					dy = y - d_y[j];
					if(dx > half_L) {
						dx -= d_L;
					} else if(dx < -half_L){
						dx += d_L;
					}
					if(dy > half_L) {
						dy -= d_L;
					} else if(dy < -half_L){
						dy += d_L;
					}
					if(dx * dx + dy * dy < diameter_square) {
						d_active[i] = 1;
						break;
					}
				}
			}
		}
	}
}
__global__ void d_make_cell_list_from_belonging_cell_list(double *d_x, double *d_y, int *d_cell_list, int *d_belonging_cell, int cell_per_axis, int N_per_cell) {
	int i, j, k, l;
	int i_global;
	int x_cell, y_cell;
	int cell_id_surrounding;

	i_global = blockDim.x * blockIdx.x + threadIdx.x;

	for(i = i_global; i < cell_per_axis * cell_per_axis; i += NUM_BLOCK * NUM_THREAD) {
		x_cell = i % cell_per_axis;
		y_cell = i / cell_per_axis;
		d_cell_list[i * N_per_cell] = 0;
		for(j = x_cell - 1; j <= x_cell + 1; j += 1) {
			for(k = y_cell - 1; k <= y_cell + 1; k += 1) {
				cell_id_surrounding = ((j + cell_per_axis) % cell_per_axis) + ((k + cell_per_axis) % cell_per_axis) * cell_per_axis;
				for(l = 0; l < d_Np; l += 1) {
					if(d_belonging_cell[l] == cell_id_surrounding) {
						d_cell_list[i * N_per_cell] += 1;
						d_cell_list[i * N_per_cell +  d_cell_list[i * N_per_cell]] = l;
					}
				}
			}
		}
		if(d_cell_list[i * N_per_cell] >= N_per_cell - 1) {
			printf("cell list overrun\n");
		}
	}
}


//------------------------------------------------------------------------------
int main(void) {	
	//variables in host
	printf("variables in host\n");
	int i;
	clock_t start, end;
	time_t whole_start, whole_end;
	int cell_per_axis;
	int N_per_cell;
	int t, t_max = 5;
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
	FILE *file_write;

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
	printf("------time%d\n", (int)(end - start));

	/*printf("----made in device global without list\n");
	start = clock();
	cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	d_check_active<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_active);
	cudaDeviceSynchronize();
	cudaMemcpy(h_check_result, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
	end = clock();
	printf("------time%d, ", (int)(end - start));
	h_DBG(h_active, h_check_result, h_Np);
	*/

	/*printf("----made in device global with list of belonging cells\n");
	start = clock();
	cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	d_make_belonging_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_cell_list, d_belonging_cell, cell_per_axis, N_per_cell);
	cudaDeviceSynchronize();
	d_check_active_with_belonging_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_active, d_belonging_cell, cell_per_axis, N_per_cell);
	cudaDeviceSynchronize();
	cudaMemcpy(h_check_result, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
	end = clock();
	printf("------time%d, ", (int)(end - start));
	h_DBG(h_active, h_check_result, h_Np);
	*/

	/*printf("----made in device global with cell list\n");
	start = clock();
	cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	d_make_belonging_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_cell_list, d_belonging_cell, cell_per_axis, N_per_cell);
	cudaDeviceSynchronize();
	d_make_cell_list_from_belonging_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_cell_list, d_belonging_cell, cell_per_axis, N_per_cell);
	cudaDeviceSynchronize();
	d_check_active_with_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_active, d_cell_list, cell_per_axis, N_per_cell);
	cudaDeviceSynchronize();
	cudaMemcpy(h_check_result, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
	end = clock();
	printf("------time%d, ", (int)(end - start));
	h_DBG(h_active, h_check_result, h_Np);
	*/
	

	/*file_write = fopen("coord.txt", "w");
	for(i = 0; i < h_Np; i += 1) {
		fprintf(file_write, "%f %f %d\n", h_x[i], h_y[i], h_active[i]);
	}
	fclose(file_write);
	 */

	printf("\ntime loop\n");
	check_renew_list = 5;
	consumed_storage_size = storage_size;
	for(t = 0; t < t_max; t += 1) {
		printf("--t=%d\n", t);
		printf("--count active particles\n");
		//check if reduction works right
		h_N_active = h_reduction_active_array(h_active, h_Np);
		//d_N_active = reduction_active_array_on_device(d_active, h_Np);
		//printf("----res_Nactive:%d\n", h_N_active - d_N_active);
		printf("----active frac:%f\n", (double)h_N_active / (double)h_Np);
		printf("----active num:%d\n", h_N_active);

		if(storage_size - consumed_storage_size < h_N_active * 6 || storage_size - consumed_storage_size < h_Np * 0.2) {
			printf("--make kick storage\n");
			gen_array_kick_on_device(gen_mt, h_kick_storage, storage_size);
			cudaDeviceSynchronize();
			consumed_storage_size = 0;
		}
		printf("--move particles\n");
		h_kick_particles(h_x, h_y, h_active, h_L, h_Np, h_N_active, h_kick_storage, consumed_storage_size);	
		consumed_storage_size += 2 * h_N_active;
		printf("----consumed kick array storage:%d\n", consumed_storage_size);

		/*if(check_renew_list == 5) {
			printf("--make new cell list\n");
			check_renew_list = 0;
			cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
			d_make_cell_list_from_belonging_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_cell_list, d_belonging_cell, cell_per_axis, N_per_cell);
			cudaDeviceSynchronize();
		}
		check_renew_list += 1;
		*/

		printf("--check activeness\n");
		printf("----check activeness on host\n");
		//check if activeness checking on device works right
		h_check_active(h_x, h_y, h_L, h_Np, h_active);

		/*printf("----check activeness on device with cell list\n");
		cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
		d_check_active_with_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_active, d_cell_list, cell_per_axis, N_per_cell);
		cudaDeviceSynchronize();
		cudaMemcpy(h_check_result, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
		printf("------");
		h_DBG(h_active, h_check_result, h_Np);
		*/

		/*printf("----check activeness on device without list\n");
		cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
		d_check_active<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_active);
		cudaDeviceSynchronize();
		cudaMemcpy(h_check_result, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_active, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
		printf("----");
		h_DBG(h_active, h_check_result, h_Np);
		*/

		/*printf("----check activeness on device global with list of belonging cells\n");
		cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
		d_make_belonging_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_cell_list, d_belonging_cell, cell_per_axis, N_per_cell);
		cudaDeviceSynchronize();
		d_check_active_with_belonging_cell_list<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_active, d_belonging_cell, cell_per_axis, N_per_cell);
		cudaDeviceSynchronize();
		cudaMemcpy(h_check_result, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
		printf("------");
		h_DBG(h_active, h_check_result, h_Np);
		*/
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
