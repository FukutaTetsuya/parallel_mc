/*
 * Cell(i, j) = cell[i + j * n]


 */
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
	int x_cell, y_cell;
	int contained_num;
	int cell_id, pair_id;
	double dx, dy;
	double dr_square;
	double diameter_square = 1.0;

	for(i = 0; i < h_Np; i += 1) {
		h_active[i] = 0;
	}
	for(i = 0; i < h_Np; i += 1) {
		x_cell = (int)(h_x[i] * (double)cell_per_axis / h_L);
		y_cell = (int)(h_y[i] * (double)cell_per_axis / h_L);

		x_cell = (x_cell + cell_per_axis) % cell_per_axis;
		y_cell = (y_cell + cell_per_axis) % cell_per_axis;

		cell_id = x_cell + y_cell * cell_per_axis;
		contained_num = h_cell_list[cell_id * N_per_cell];

		for(j = 1; j < contained_num; j += 1) {
			pair_id = h_cell_list[cell_id * N_per_cell + j];
			dx = h_x[i] - h_x[pair_id];
			if(dx > 0.5 * h_L) {
				dx -= h_L;
			} else if(dx < -0.5 * h_L) {
				dx += h_L;
			}
			dy = h_y[i] - h_y[pair_id];
			if(dy > 0.5 * h_L) {
				dy -= h_L;
			} else if(dy < -0.5 * h_L) {
				dy += h_L;
			}

			dr_square = dx * dx + dy * dy;
			if(dr_square < diameter_square) {
				h_active[i] = 1;
				break;
			}
		}

/*		for(j = 0; j < i; j += 1) {
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
*/
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

int h_make_cell_list(double *h_x, double *h_y, double h_L, int h_Np, int *h_cell_list, int cell_per_axis, int N_per_cell) {
	//h_make_cell_list(h_x, h_y, h_L, h_Np, h_cell_list, cell_per_axis, N_per_cell);
	int i, j, k;
	int x_cell, y_cell;
	int cell_id;
	int cell_list_size = cell_per_axis * cell_per_axis * N_per_cell;
	int contained_num;
	FILE *file;
	//init cell list
	for(i = 0; i < cell_list_size; i += 1) {
		h_cell_list[i] = 0;
	}
	file = fopen("cell_id.txt", "w");
	//make cell list
	for(i = 0; i < h_Np; i += 1) {
		x_cell = (int)(h_x[i] * (double)cell_per_axis / h_L);
		y_cell = (int)(h_y[i] * (double)cell_per_axis / h_L);
		fprintf(file, "%d %d\n", i, ((x_cell + cell_per_axis) % cell_per_axis) + ((y_cell + cell_per_axis) % cell_per_axis) * cell_per_axis);
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
	fclose(file);
	return 0;
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
__global__ void make_cell_list(double *d_x, double *d_y, double d_L, int d_Np, int *d_cell_list) {
}

//------------------------------------------------------------------------------
int main(void) {
	int i;
	clock_t start, end;
	int cell_per_axis;
	int N_per_cell;
	FILE *file;

	//variables in host
	double *h_x;
	double *h_y;
	double h_L;
	int *h_active;
	int *h_check_result;
	int h_Np;
	int *h_cell_list;
	int *h_active_DBG;

	//variables in device
	double *d_x;
	double *d_y;
	int *d_active;
	int *d_cell_list;

	//initialize
	init_genrand(19970303);

	//--set variable
	h_Np = 18000;
	h_L = 140.0;
	cell_per_axis = (int)(h_L / 11.0) + 1;//renew list every 5 steps
	N_per_cell = (h_Np * 13) / (cell_per_axis * cell_per_axis);
	printf("cell per axis:%d N_per_cell:%d\n", cell_per_axis, N_per_cell);

	cudaMemcpyToSymbol(d_Np, &h_Np, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_L, &h_L, sizeof(double), 0, cudaMemcpyHostToDevice);

	//--allocate memory
	cudaHostAlloc((void **)&h_x, h_Np * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_y, h_Np * sizeof(double), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_active, h_Np * sizeof(int), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_check_result, h_Np * sizeof(int), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_cell_list, cell_per_axis * cell_per_axis * N_per_cell * sizeof(int), cudaHostAllocMapped);
	h_active_DBG = (int *)calloc(h_Np, sizeof(int));

	cudaMalloc((void **)&d_x, h_Np * sizeof(double));
	cudaMalloc((void **)&d_y, h_Np * sizeof(double));
	cudaMalloc((void **)&d_active, h_Np * sizeof(int));
	cudaMalloc((void **)&d_cell_list, cell_per_axis * cell_per_axis * N_per_cell * sizeof(int));

	//--place particles
	init_configuration(h_x, h_y, h_L, h_Np);
	cudaMemcpy(d_x, h_x, h_Np * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, h_Np * sizeof(double), cudaMemcpyHostToDevice);

	//--make first acriveness array
	//----made in host
	start = clock();
	h_check_active(h_x, h_y, h_L, h_Np, h_active);
	end = clock();
	printf("straighforward:%d [ms]\n", (int)((end - start)*1000 /CLOCKS_PER_SEC ));

	//----made in host with cell list
	start = clock();
	h_make_cell_list(h_x, h_y, h_L, h_Np, h_cell_list, cell_per_axis, N_per_cell);
	h_check_active_with_list(h_x, h_y, h_L, h_Np, h_active_DBG, h_cell_list, cell_per_axis, N_per_cell);
	end = clock();
	file = fopen("init_cood.txt", "w");
	for(i = 0; i < h_Np; i += 1) {
		fprintf(file, "%d %f %f\n", i, h_x[i], h_y[i]);
	}
	fclose(file);
	file = fopen("cell_list.txt", "w");
	for(i = 0; i < N_per_cell * cell_per_axis * cell_per_axis; i += 1) {
		fprintf(file, "%d %d\n", i, h_cell_list[i]);
	}
	fclose(file);
	printf("cell list:%d [ms]\n", (int)((end - start)*1000 /CLOCKS_PER_SEC ));

	//----made in device global
	start = clock();
	d_check_active<<<NUM_BLOCK, NUM_THREAD>>>(d_x, d_y, d_active);
	cudaDeviceSynchronize();
	cudaMemcpy(h_check_result, d_active, h_Np * sizeof(int), cudaMemcpyDeviceToHost);
	end = clock();
	printf("gpu:%d [ms]\n", (int)((end - start)*1000 /CLOCKS_PER_SEC ));
	
	printf("\n");
	for(i = 0; i < cell_per_axis * cell_per_axis; i += 1) {
		printf("(%d,%d) ", i, h_cell_list[i * N_per_cell]);
	}
	printf("\n");
	printf("gpu:");
	h_DBG(h_active, h_check_result, h_Np);
	printf("cell_list:");
	h_DBG(h_active, h_active_DBG, h_Np);

	//move particles

	//finalize
	//--free memory
       	cudaFreeHost(h_x);
	cudaFreeHost(h_y);
	cudaFreeHost(h_active);
	cudaFreeHost(h_check_result);
	cudaFreeHost(h_cell_list);
	free(h_active_DBG);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_active);
	cudaFree(d_cell_list);
	return 0;
}
