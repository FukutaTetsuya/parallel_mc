/*inculde files-----------------------------------------------------------------*/
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<curand.h>
#include"mt.h"
/*constants--------------------------------------------------------------------------*/
__device__ __constant__ int d_Np;
__device__ __constant__ double d_L;
#define NUM_BLOCK 5
#define NUM_THREAD 1024
#define PI 3.1415926535897932384626433
/*predeclaration-----------------------------------------------------------------------------------*/
__global__ void d_check_active(double *d_x, double *d_y, int *d_active);
__global__ void reduce_array_shared_memory(int *array, int *array_reduced, int dim_array);
__global__ void d_check_active_with_cell_list(double *d_x, double *d_y, int *d_active, int *d_cell_list, int cell_per_axis, int N_per_cell);
__global__ void d_make_belonging_cell_list(double *d_x, double *d_y, int *d_cell_list, int *d_belonging_cell, int cell_per_axis, int N_per_cell);
__global__ void d_check_active_with_belonging_cell_list(double *d_x, double *d_y, int *d_active, int *d_belonging_cell, int cell_per_axis, int N_per_cell);
__global__ void d_make_cell_list_from_belonging_cell_list(double *d_x, double *d_y, int *d_cell_list, int *d_belonging_cell, int cell_per_axis, int N_per_cell);
int reduction_active_array_on_device(int *d_active, int h_Np);
int h_kick_particles(double *h_x, double *h_y, int *h_active, double h_L, int h_Np, int h_N_active, float *h_kick_storage, int consumed_storage_size);
void gen_array_kick_on_device(curandGenerator_t gen_mt, float *h_kick_storage, size_t storage_size);
void init_configuration(double *h_x, double *h_y, double h_L, int h_Np);

/*declaration-----------------------------------------------------------------------------------*/

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

int h_kick_particles(double *h_x, double *h_y, int *h_active, double h_L, int h_Np, int h_N_active, float *h_kick_storage, int consumed_storage_size) {
	int i;
	int num = 0;

	for(i = 0; i < h_Np; i += 1) {
		if(h_active[i] == 1) {
			h_x[i] += h_kick_storage[2 * i + consumed_storage_size];
			if(h_x[i] < 0) {
				h_x[i] += h_L;
			} else if(h_x[i] > h_L) {
				h_x[i] -= h_L;
			}
			h_y[i] += h_kick_storage[2 * i + 1 + consumed_storage_size];
			if(h_y[i] < 0) {
				h_y[i] += h_L;
			} else if(h_y[i] > h_L) {
				h_y[i] -= h_L;
			}
			num += 1;
		}
	}
	return num;
}
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


void gen_array_kick_on_device(curandGenerator_t gen_mt, float *h_kick_storage, size_t storage_size) {
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
void init_configuration(double *h_x, double *h_y, double h_L, int h_Np) {
	int i;

	for(i = 0; i < h_Np; i += 1) {
		h_x[i] = h_L * genrand_real2();
		h_y[i] = h_L * genrand_real2();
	}
}

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

