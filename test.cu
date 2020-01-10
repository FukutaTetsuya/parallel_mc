#include<stdio.h>
#include<time.h>
#include"mt.h"

#define NUM_THREAD 1024
__device__ __constant__ int d_Np;

__global__ void reduce_array_shared_memory(int *array, int *array_reduced, int dim_array, int num_block) {
	__shared__ int array_shared[NUM_THREAD];
	int global_id = threadIdx.x + blockIdx.x * blockDim.x;
	int block_id = blockIdx.x;
	int local_id = threadIdx.x;
	int i, j;
	for(i = global_id; i < dim_array; i += num_block * NUM_THREAD) {
		array_shared[local_id] = array[i];
		__syncthreads();

		//このj多分間違い
		j = NUM_THREAD;
		while(j > 1) {
			if(j % 2 == 0) {
				j = j / 2;
			} else {
				j = 1 + j / 2;
			}
			__syncthreads();
		}

		if(local_id == 0) {
			array_reduced[block_id] = array_shared[0];
		}
		__syncthreads();
		block_id += num_block;
	}
}

void fill_array(int *array, int dim) {
	int i;
	for(i = 0; i < dim; i += 1) {
		if(genrand_real1() < 0.5) {
			array[i] = 0;
		} else {
			array[i] = 1;
		}
	}
}

int h_count_active_particle(int *h_active, int h_Np) {
	int i;
	int sum;
	sum = 0;
	for(i = 0; i < h_Np; i += 1) {
		sum += h_active[i];
	}
	return sum;
}

int count_active_on_device(int *d_active, int h_Np, int num_block) {
	int i, j, k;
	int i_temp;
	int *d_reduction[2];
	int h_answer;
	cudaMalloc((void **)&d_reduction[0], h_Np * sizeof(int));
	cudaMalloc((void **)&d_reduction[1], h_Np * sizeof(int));
	cudaMemcpy(d_reduction[0], d_active, h_Np * sizeof(int), cudaMemcpyDeviceToDevice);
	i = 0;
	j = 1;
	for(k = h_Np; k > 1; k = k / NUM_THREAD) {
		reduce_array_shared_memory<<<num_block, NUM_THREAD>>>(d_reduction[i], d_reduction[j], k, num_block);
		cudaDeviceSynchronize();
		i_temp = i;
		i = j;
		j = i_temp;
	}
	cudaMemcpy(&h_answer, d_reduction[i], sizeof(int), cudaMemcpyDeviceToHost);
	printf("gpu:%d\n", h_answer);

	cudaFree(d_reduction[0]);
	cudaFree(d_reduction[1]);
	return 0;
}

int main(void){
	int *h_active;
	int *d_active;
	int h_Np = 2000;
	int h_ans, d_ans;
	int num_block = h_Np / NUM_THREAD + 1;

	cudaMalloc((void **)&d_active, h_Np * sizeof(int));
	cudaHostAlloc((void **)&h_active, h_Np * sizeof(int), cudaHostAllocMapped);
	cudaMemcpyToSymbol(d_Np, &h_Np, sizeof(int), 0, cudaMemcpyHostToDevice);
	init_genrand((int)time(NULL));

	fill_array(h_active, h_Np);
	cudaMemcpy(d_active, h_active, h_Np * sizeof(int), cudaMemcpyHostToDevice);
	h_ans = h_count_active_particle(h_active, h_Np);
	printf("cpu:%d\n", h_ans);

	count_active_on_device(d_active, h_Np, num_block);

	cudaFree(d_active);
	cudaFreeHost(h_active);

	return 0;
}
