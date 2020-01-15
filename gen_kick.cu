#include<stdio.h>
#include<time.h>
#include<math.h>
#include<curand.h>
#include<cuda.h>

#define NUM_BLOCK 5
#define NUM_THREAD 1024

__device__ __constant__ int d_Np;

__global__ void show_inside_d_array_kick(int d_N_active, float *d_array_kick) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float x, y;

	for(; i < d_N_active; i += NUM_BLOCK + NUM_THREAD) {
		x = d_array_kick[i * 2];
		y = d_array_kick[i * 2 + 1];
		if(i < 10) {
			printf("%d:%f,%f\n", i, x, y);
		}
	}
}

void gen_array_kick(curandGenerator_t gen_mt, int d_N_active, float *d_array_kick, float *h_array_kick) {
	float *d_array_temp;
	float *h_array_temp;
	size_t n = d_N_active * 2;
	int i, j;
	int if_break;
	float x, y;
	double l;

	if(cudaSuccess != cudaMalloc((void **)&d_array_temp, n * sizeof(float))) {
		printf("failed to alloc on device\n");
	}
	h_array_temp = (float *)calloc(n , sizeof(float));
	if(h_array_temp == NULL) {
		printf("failed to alloc on host\n");
	}

	i = 0;
	if_break = 0;
	while(i < n) {
		curandGenerateUniform(gen_mt, d_array_temp, n);
		cudaDeviceSynchronize();
		cudaMemcpy(h_array_temp, d_array_temp, n * sizeof(float), cudaMemcpyDeviceToHost);

		for(j = 0; j < n; j += 2) {
			x = 1.0 - 2.0 * h_array_temp[j];
			y = 1.0 - 2.0 * h_array_temp[j + 1];
			l = x * x + y * y;
			if(l <= 1.0) {
				l = sqrt(l);
				h_array_kick[i] = x * l * 0.5;
				h_array_kick[i + 1] = y * l * 0.5;
				i += 2;
			}
			if(i >= n - 1) {
				if_break = 1;
				break;
			}
		}
		if(if_break == 1) {
			break;
		}
	}

	cudaMemcpy(d_array_kick, h_array_kick, n * sizeof(float), cudaMemcpyHostToDevice);

	cudaFree(d_array_temp);
	free(h_array_temp);
}

int main(void) {
	float *d_array_kick;
	float *h_array_kick;
	int d_N_active = 5240;//indeed it's on host
	int n = d_N_active * 2;
	curandGenerator_t gen_mt;
	
	cudaMalloc((void **)&d_array_kick, n * sizeof(float));
	h_array_kick = (float *)calloc(n, sizeof(float));

	//set up generator
	curandCreateGenerator(&gen_mt, CURAND_RNG_PSEUDO_MTGP32);
	//curandSetPseudoRandomGeneratorSeed(gen_mt, 19970303ULL);
	curandSetPseudoRandomGeneratorSeed(gen_mt, (unsigned long)time(NULL));
	curandSetGeneratorOffset(gen_mt, 0ULL);
	curandSetGeneratorOrdering(gen_mt, CURAND_ORDERING_PSEUDO_DEFAULT);
	printf("set-ed gene\n");

	//make d_N_active random kick 
	gen_array_kick(gen_mt, d_N_active, d_array_kick, h_array_kick);
	cudaDeviceSynchronize();
	printf("gen-ed kick array\n");

	show_inside_d_array_kick<<<NUM_BLOCK, NUM_THREAD>>>(d_N_active, d_array_kick);

	//finalize
	curandDestroyGenerator(gen_mt);
	cudaFree(d_array_kick);
	free(h_array_kick);
	printf("finalized\n");
	return 0;
}
