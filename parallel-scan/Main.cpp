#include <stdlib.h>
#include <stdio.h>

#include <time.h>

#include "scan.cuh"
#include "utils.h"

void test(int N) {
	bool canBeBlockscanned = N <= 1024;

	time_t t;
	srand((unsigned)time(&t));
	int *in = new int[N];
	for (int i = 0; i < N; i++) {
		in[i] = rand() % 10;
	}

	printf("%i Elements \n", N);

		// sequential scan on CPU
		int *outHost = new int[N]();
		long time_host = sequential_scan(outHost, in, N);
		printResult("host    ", outHost[N - 1], time_host);

		// full scan
		int *outGPU = new int[N]();
		float time_gpu = scan(outGPU, in, N, false);
		printResult("gpu     ", outGPU[N - 1], time_gpu);
	
		// full scan with BCAO
		int *outGPU_bcao = new int[N]();
		float time_gpu_bcao = scan(outGPU_bcao, in, N, true);
		printResult("gpu bcao", outGPU_bcao[N - 1], time_gpu_bcao);

		if (canBeBlockscanned) {
			// basic level 1 block scan
			int *out_1block = new int[N]();
			float time_1block = blockscan(out_1block, in, N, false);
			printResult("level 1 ", out_1block[N - 1], time_1block);

			// level 1 block scan with BCAO
			int *out_1block_bcao = new int[N]();
			float time_1block_bcao = blockscan(out_1block_bcao, in, N, true);
			printResult("l1 bcao ", out_1block_bcao[N - 1], time_1block_bcao);

			delete[] out_1block;
			delete[] out_1block_bcao;
		}

	printf("\n");

	delete[] in;
	delete[] outHost;
	delete[] outGPU;
	delete[] outGPU_bcao;
}

int main()
{
	int TEN_MILLION = 10000000;
	int ONE_MILLION = 1000000;
	int TEN_THOUSAND = 10000;

	int elements[] = {
		TEN_MILLION * 2,
		TEN_MILLION,
		ONE_MILLION,
		TEN_THOUSAND,
		5000,
		4096,
		2048,
		2000,
		1000,
		500,
		100,
		64,
		8,
		5
	};

	int numElements = sizeof(elements) / sizeof(elements[0]);

	for (int i = 0; i < numElements; i++) {
		test(elements[i]);
	}

	return 0;
}