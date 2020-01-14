/*
   *mariの
   *curandSetPseudoRandomGeneratorSeed(mari, (unsigned long)time(NULL));
   *が反映されていない？
   *同じ値が帰ってくる
   */
#include<stdio.h>
#include<time.h>
#include<curand.h>
#include<cuda.h>

int main(void) {
	float *d_array_rei;
	float *d_array_mari;
	float *h_array_rei;
	float *h_array_mari;
	size_t n = 16384;
	unsigned int dimension = 2;
	float mean = 0.0;
	float standard_deviation = 1.0;
	int i;
	//これが乱数生成器を指す名前的なもの
	//二つ作ってみる
	curandGenerator_t rei, mari;
	
	//デバイスとホストにメモリを確保する
	cudaMalloc((void **)&d_array_rei, n * sizeof(float));
	cudaMalloc((void **)&d_array_mari, n * sizeof(float));
	cudaHostAlloc((void **)&h_array_rei, n * sizeof(float), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_array_mari, n * sizeof(float), cudaHostAllocMapped);

	//乱数生成器を作る
	//\->reiはXORWOWというアルゴリズムを使い擬似乱数を作る乱数生成器とする
	curandCreateGenerator(&rei, CURAND_RNG_PSEUDO_XORWOW);
	//\->mariはSOBOLというアルゴリズムを使い準乱数を作る乱数生成器とする
	curandCreateGenerator(&mari, CURAND_RNG_QUASI_SOBOL32);

	//乱数生成器にシードを与える。ULLは型、64bit符号なし整数
	curandSetPseudoRandomGeneratorSeed(rei, 890106ULL);
	curandSetPseudoRandomGeneratorSeed(mari, 890106ULL);
	//\->time()を使うならこちら
	curandSetPseudoRandomGeneratorSeed(rei, (unsigned long)time(NULL));
	curandSetPseudoRandomGeneratorSeed(mari, (unsigned long)time(NULL));

	//オフセットを伝える。これも64bit符号なし整数で指定する
	//\->reiにだけオフセットを設け、mariはオフセットなしとする
	curandSetGeneratorOffset(rei, 5ULL);

	//rei,mariに格納順を伝える、どちらもデフォルトでよかろう
	curandSetGeneratorOrdering(rei, CURAND_ORDERING_PSEUDO_DEFAULT);
	curandSetGeneratorOrdering(mari, CURAND_ORDERING_QUASI_DEFAULT);

	//準乱数については、何次元空間で均一に分布するかを指定できる
	curandSetQuasiRandomGeneratorDimensions(mari, dimension);

	//n個だけ乱数を作らせ、結果をd_arrayに収める
	//\->reiにはfloatの一様乱数をつくらせる
	curandGenerateUniform(rei, d_array_rei, n);
	//\->mariにはfloatの正規分布乱数をつくらせる
	curandGenerateNormal(mari, d_array_mari, n, mean, standard_deviation);

	//デバイスからホストへ生成された乱数を持ってくる
	cudaMemcpy(h_array_rei, d_array_rei, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_array_mari, d_array_mari, n * sizeof(float), cudaMemcpyDeviceToHost);

	printf("rei's result\n");
	for(i = 0; i < 10; i += 1) {
		printf("%d:%f\n", i, h_array_rei[i]);
	}
	printf("mari's result\n");
	for(i = 0; i < 10; i += 1) {
		printf("%d:%f\n", i, h_array_mari[i]);
	}

	//乱数生成器を消す
	curandDestroyGenerator(rei);
	curandDestroyGenerator(mari);
	//デバイスとホストのメモリを解放する
	cudaFree(d_array_rei);
	cudaFree(d_array_mari);
	cudaFreeHost(h_array_rei);
	cudaFreeHost(h_array_mari);
	return 0;
}
