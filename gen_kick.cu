#include<stdio.h>
#include<time.h>
#include<math.h>
#include<curand.h>
#include<cuda.h>

#define NUM_BLOCK 5
#define NUM_THREAD 1024

__device__ __constant__ int d_Np;

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
				printf("broke at ending_point:%d\n", ending_point);
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

void make_histogram(double max_value, float *array, int dim_array) {
	int i, j;
	double delta;
	double dx, dy, dr;
	int histogram[100];
	int place;
	delta = max_value / 100;
	for(i = 0; i < 100; i += 1) {
		histogram[i] = 0;
	}

	for(i = 0; i < dim_array; i += 2) {
		dx = array[i];
		dy = array[i + 1];
		dr = sqrt(dx * dx + dy * dy);
		place = (int)(dr / delta);
		if(place < 99) {
			histogram[place] += 1;
		} else {
			histogram[99] += 1;
		}
	}
	for(i = 0; i < 100; i += 1) {
		printf("%d:%d\n", i, histogram[i]);
	}
}

int main(void) {
	double dx, dy, dr;
	float *h_kick_storage;
	size_t storage_size = 1600000;
	curandGenerator_t gen_mt;
	
	h_kick_storage = (float *)calloc(storage_size, sizeof(float));
	if(h_kick_storage == NULL) {
		printf("failed to allocate h_kick_storage\n");
		return 0;
	}

	//set up generator
	curandCreateGenerator(&gen_mt, CURAND_RNG_PSEUDO_MTGP32);
	//curandSetPseudoRandomGeneratorSeed(gen_mt, 19970303ULL);
	curandSetPseudoRandomGeneratorSeed(gen_mt, (unsigned long)time(NULL));
	curandSetGeneratorOffset(gen_mt, 0ULL);
	curandSetGeneratorOrdering(gen_mt, CURAND_ORDERING_PSEUDO_DEFAULT);
	printf("set-ed generator\n");

	//make d_N_active random kick 
	gen_array_kick(gen_mt, h_kick_storage, storage_size);
	cudaDeviceSynchronize();
	printf("gen-ed kick array\n");

	dx = h_kick_storage[0];
	dy = h_kick_storage[1];
	dr = sqrt(dx * dx + dy * dy);
	printf("%f %f %f\n", dx ,dy ,dr);
	dx = h_kick_storage[storage_size - 2];
	dy = h_kick_storage[storage_size - 1];
	dr = sqrt(dx * dx + dy * dy);
	printf("%f %f %f\n", dx ,dy ,dr);


	make_histogram(0.5, h_kick_storage, storage_size);


	//finalize
	curandDestroyGenerator(gen_mt);
	free(h_kick_storage);
	printf("finalized\n");
	return 0;
}
