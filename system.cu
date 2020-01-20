/*
 * Cell(i, j) = cell[i + j * n]
 */
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

//pre declaration---------------------------------------------------------------
__global__ void reduce_array_shared_memory(int *array, int *array_reduced, int dim_array);

//host functions----------------------------------------------------------------
void init_configuration(double *h_x, double *h_y, double h_L, int h_Np) {
	int i;

	for(i = 0; i < h_Np; i += 1) {
		h_x[i] = h_L * genrand_real2();
		h_y[i] = h_L * genrand_real2();
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

int h_reduction_active_array(int *h_active, int h_Np) {
	int i;
	int sum;
	sum = 0;
	for(i = 0; i < h_Np; i += 1) {
		sum += h_active[i];
	}
	return sum;
}

int reduction_active_array_on_device(int *d_active, int h_Np) {
	int i, j, k;
	int i_temp;
	int *d_reduction[2];
	int h_answer;
	cudaMalloc((void **)&d_reduction[0], h_Np * sizeof(int));
	cudaMalloc((void **)&d_reduction[1], h_Np * sizeof(int));
	i = 0;
	j = 1;
	cudaMemcpy(d_reduction[i], d_active, h_Np * sizeof(int), cudaMemcpyDeviceToDevice);
	for(k = h_Np; k > 1; k = 1 + k / NUM_THREAD) {
		reduce_array_shared_memory<<<NUM_BLOCK, NUM_THREAD>>>(d_reduction[i], d_reduction[j], k);
		cudaDeviceSynchronize();
		i_temp = i;
		i = j;
		j = i_temp;
	}
	cudaMemcpy(&h_answer, d_reduction[i], sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_reduction[0]);
	cudaFree(d_reduction[1]);
	return h_answer;
}


void h_DBG(int *A, int *B, int dim) {
	int i;
	double res = 0.0;
	for(i = 0; i < dim; i += 1) {
		res += (A[i] - B[i]) * (A[i] - B[i]);
	}
	printf("res %f\n", res);
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
	return 0;
}

void gen_array_kick(curandGenerator_t gen_mt, float *h_kick_storage, size_t storage_size) {
	float *d_array_temp;
	float *h_array_temp;
	float *h_array_kick;
	size_t generate_size = 16384;
	int i, j;
	int if_break;
	int starting_point, ending_point;
	float x, y;
	double l;

	if(cudaSuccess != cudaMalloc((void **)&d_array_temp, generate_size * sizeof(float))) {
		printf("failed to alloc on device\n");
	}
	h_array_temp = (float *)calloc(generate_size , sizeof(float));
	h_array_kick = (float *)calloc(generate_size , sizeof(float));
	if(h_array_temp == NULL || h_array_kick == NULL) {
		printf("failed to alloc on host\n");
	}

	i = 0;
	if_break = 0;
	starting_point = 0;
	ending_point= 0;

	while(i < storage_size) {
		curandGenerateUniform(gen_mt, d_array_temp, generate_size);
		cudaDeviceSynchronize();
		cudaMemcpy(h_array_temp, d_array_temp, generate_size * sizeof(float), cudaMemcpyDeviceToHost);

		starting_point = ending_point;
		i = 0;
		for(j = 0; j < generate_size; j += 2) {
			x = 1.0 - 2.0 * h_array_temp[j];
			y = 1.0 - 2.0 * h_array_temp[j + 1];
			l = x * x + y * y;
			if(l <= 1.0) {
				l = sqrt(l);
				h_array_kick[i] = x * l * 0.5;
				h_array_kick[i + 1] = y * l * 0.5;
				i += 2;
				ending_point += 2;
			}
			if(ending_point >= storage_size - 1) {
				if_break = 1;
				//printf("broke at ending_point:%d\n", ending_point);
				break;
			}
		}
		memcpy(&h_kick_storage[starting_point], h_array_kick, i * sizeof(float));
		if(if_break == 1) {
			break;
		}
	}


	cudaFree(d_array_temp);
	free(h_array_temp);
	free(h_array_kick);
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

__global__ void d_check_active_with_list(double *d_x, double *d_y, int *d_active, int *d_cell_list, int cell_per_axis, int N_per_cell) {
	//d_L and d_Np are already declared as __global__ const
	int j;
	int x_c, y_c;
	int cell_id, N_in_cell;
	int pair_id;
	int i_global;
	double dx, dy, dr_square;
	double diameter_square = 1.0;
	i_global = blockDim.x * blockIdx.x + threadIdx.x;
	if(i_global < d_Np) {
		d_active[i_global] = 0;
		x_c = (int)(d_x[i_global] * (double)cell_per_axis / d_L);
		y_c = (int)(d_y[i_global] * (double)cell_per_axis / d_L);
		cell_id = x_c + y_c * cell_per_axis;
		N_in_cell = d_cell_list[cell_id * N_per_cell];	
		for(j = 1; j <= N_in_cell; j += 1) {
			pair_id = d_cell_list[cell_id * N_per_cell + j];
			if(i_global == pair_id) {continue;}
			dx = d_x[i_global] - d_x[pair_id];
			dy = d_y[i_global] - d_y[pair_id];
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
				d_active[i_global] = 1;
			}
		}
	}
}

__global__ void d_make_cell_list(double *d_x, double *d_y, int *d_cell_list, int *d_belonging_cell, int cell_per_axis, int N_per_cell) {
	//this func needs equal to or more than Np threads
	int j, k, l;
	int i_global;
	int x_cell, y_cell;
	int cell_id;

	i_global = blockDim.x * blockIdx.x + threadIdx.x;
	if(i_global < d_Np) {
		x_cell = (int)(d_x[i_global] * (double)cell_per_axis / d_L);
		y_cell = (int)(d_y[i_global] * (double)cell_per_axis / d_L);
		cell_id = x_cell + y_cell * cell_per_axis;
		d_belonging_cell[i_global] = cell_id;
	}
	__syncthreads();
	if(i_global < cell_per_axis * cell_per_axis) {
		d_cell_list[i_global * N_per_cell] = 0;
		x_cell = i_global % cell_per_axis;
		y_cell = i_global / cell_per_axis;
		for(j = x_cell - 1; j <= x_cell + 1; j += 1) {
			for(k = y_cell - 1; k <= y_cell + 1; k += 1) {
				cell_id = ((j + cell_per_axis) % cell_per_axis) + ((k + cell_per_axis) % cell_per_axis) * cell_per_axis;
				for(l = 0; l < d_Np; l += 1) {
					if(d_belonging_cell[l] == cell_id) {
						d_cell_list[i_global * N_per_cell] += 1;
						d_cell_list[i_global * N_per_cell +  d_cell_list[i_global * N_per_cell] ] = l;
					}
				}
			}
		}
	}
}

__global__ void reduce_array_shared_memory(int *array, int *array_reduced, int dim_array) {
	__shared__ int array_shared[NUM_THREAD];
	int global_id = threadIdx.x + blockIdx.x * blockDim.x;
	int block_id = blockIdx.x;
	int local_id = threadIdx.x;
	int i, j;
	int iterate_max = 1 + dim_array / (NUM_THREAD * NUM_BLOCK);
	int iterate;

	for(iterate = 0; iterate < iterate_max; iterate += 1) {
		i = global_id + iterate * NUM_BLOCK * NUM_THREAD;
		if(i < d_Np) {
			array_shared[local_id] = array[i];
		} else {
			array_shared[local_id] = 0;
		}
		__syncthreads();

		for(j = NUM_THREAD / 2; j > 0; j /= 2) {
			if((local_id < j) && (local_id + j < dim_array)) {
				array_shared[local_id] += array_shared[local_id + j]; 
			}
		__syncthreads();
		}

		if(local_id == 0) {
			array_reduced[block_id] = array_shared[0];
		}
		__syncthreads();
		block_id += NUM_BLOCK;
	}
}

//------------------------------------------------------------------------------
int main(void) {
	clock_t start, end;
	int cell_per_axis;
	int N_per_cell;

	//variables in host
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
	size_t storage_size = 1600000;
	curandGenerator_t gen_mt;

	//variables in device
	double *d_x;
	double *d_y;
	int *d_active;
	int *d_cell_list;
	int *d_belonging_cell;
	int d_N_active;

	//initialize
	//--set up random number generators

	//----host mt
	//init_genrand(19970303);
	init_genrand((unsigned int)time(NULL));
	//----cuRAND host API
	curandCreateGenerator(&gen_mt, CURAND_RNG_PSEUDO_MTGP32);
	//curandSetPseudoRandomGeneratorSeed(gen_mt, (unsigned long)time(NULL));
	curandSetPseudoRandomGeneratorSeed(gen_mt, (unsigned long)19970303);
	curandSetGeneratorOffset(gen_mt, 0ULL);
	curandSetGeneratorOrdering(gen_mt, CURAND_ORDERING_PSEUDO_DEFAULT);

	//--set variable
	h_Np = 18000;
	h_L = 140.0;
	cell_per_axis = (int)(h_L / 11.0) + 1;//renew list every 5 steps
	N_per_cell = (h_Np * 13) / (cell_per_axis * cell_per_axis);
	printf("cell per axis:%d N_per_cell:%d\n", cell_per_axis, N_per_cell);

	cudaMemcpyToSymbol(d_Np, &h_Np, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_L, &h_L, sizeof(double), 0, cudaMemcpyHostToDevice);

	//--allocate memory
	//----memory on host
	cudaHostAlloc((void **)&h_x, h_Np * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_y, h_Np * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_active, h_Np * sizeof(int), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_check_result, h_Np * sizeof(int), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_cell_list, cell_per_axis * cell_per_axis * N_per_cell * sizeof(int), cudaHostAllocMapped);
	h_active_DBG = (int *)calloc(h_Np, sizeof(int));
	h_kick_storage = (float *)calloc(storage_size, sizeof(float));

	//----memory on device
	cudaMalloc((void **)&d_x, h_Np * sizeof(double));
	cudaMalloc((void **)&d_y, h_Np * sizeof(double));
	cudaMalloc((void **)&d_active, h_Np * sizeof(int));
	cudaMalloc((void **)&d_cell_list, cell_per_axis * cell_per_axis * N_per_cell * sizeof(int));
	cudaMalloc((void **)&d_belonging_cell, h_Np * sizeof(int));

	//--place particles
	init_configuration(h_x, h_y, h_L, h_Np);
	cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);

	//--make first acriveness array
	//----made in host
	start = clock();
	h_check_active(h_x, h_y, h_L, h_Np, h_active);
	end = clock();
	//printf("straighforward:%d [ms]\n\n", (int)((end - start)*1000 /CLOCKS_PER_SEC ));
	printf("straighforward:%d\n\n", (int)(end - start));

	//----made in host with cell list
	start = clock();
	h_make_cell_list(h_x, h_y, h_L, h_Np, h_cell_list, cell_per_axis, N_per_cell);
	h_check_active_with_list(h_x, h_y, h_L, h_Np, h_active_DBG, h_cell_list, cell_per_axis, N_per_cell);
	end = clock();
	printf("host cell list:%d\n", (int)(end - start));
	h_DBG(h_active, h_active_DBG, h_Np);
	printf("\n");

	//----made in device global
	start = clock();
	cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	d_check_active<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_active);
	cudaDeviceSynchronize();
	cudaMemcpy(h_check_result, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
	end = clock();
	printf("gpu:%d\n", (int)(end - start));
	h_DBG(h_active, h_check_result, h_Np);
	printf("\n");

	//----made in device global with list, list is made in device
	start = clock();
	cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	d_make_cell_list<<<1, h_Np>>>(d_x, d_y, d_cell_list, d_belonging_cell, cell_per_axis, N_per_cell);
	cudaDeviceSynchronize();
	d_check_active_with_list<<<1, h_Np>>>(d_x, d_y, d_active, d_cell_list, cell_per_axis, N_per_cell);
	cudaDeviceSynchronize();
	cudaMemcpy(h_check_result, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
	end = clock();
	printf("gpu with gpu list:%d\n", (int)(end - start));
	h_DBG(h_active, h_check_result, h_Np);
	printf("\n");

	//time loop
	//--move particles
	gen_array_kick(gen_mt, h_kick_storage, storage_size);
	cudaDeviceSynchronize();
	//--check activeness
	//--count active particles
	h_N_active = h_reduction_active_array(h_active, h_Np);
	d_N_active = reduction_active_array_on_device(d_active, h_Np);
	printf("res_Nactive:%d\n", h_N_active - d_N_active);
	printf("active frac:%f\n", (double)d_N_active / (double)h_Np);
	//--(sometimes) make new cell list

	//finalize
	//--free memory
	cudaFreeHost(h_x);
	cudaFreeHost(h_y);
	cudaFreeHost(h_active);
	cudaFreeHost(h_check_result);
	cudaFreeHost(h_cell_list);
	free(h_active_DBG);
	free(h_kick_storage);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_active);
	cudaFree(d_cell_list);
	cudaFree(d_belonging_cell);
	//--destroy random number generator
	curandDestroyGenerator(gen_mt);
	return 0;
}
