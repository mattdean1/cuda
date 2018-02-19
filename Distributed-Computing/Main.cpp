#include <stdlib.h>
#include <stdio.h>

#include <time.h>

#include "scan.cuh"
#include "helpers.h"

void test(int N) {
	int *in = new int[N];
	int *outDevice = new int[N]();
	int *outHost = new int[N]();

	for (int i = 0; i < N; i++) {
		in[i] = rand() % 10;
	}

	scan(outDevice, in, N);
	sequential_scan(outHost, in, N);

	printResult("input length", N);
	printResult("device", outDevice[N - 1]);
	printResult("host  ", outHost[N - 1]);
	printf("\n");

	delete[] in;
	delete[] outDevice;
	delete[] outHost;
}




void test_blockscan(int N) {
	int *outDevice = new int[N]();
	int *outDeviceBcao = new int[N]();
	int *outHost = new int[N]();

	int *in = new int[N];
	for (int i = 0; i < N; i++) {
		in[i] = rand() % 10;
	}

	// dummy call to "warm up" the GPU
	blockscan(outDevice, in, N, false);


	long start = get_nanos();
	blockscan(outDevice, in, N, false);
	long end = get_nanos();

	long start_bcao = get_nanos();
	blockscan(outDeviceBcao, in, N, true);
	long end_bcao = get_nanos();
	
	long start_host = get_nanos();
	sequential_scan(outHost, in, N);
	long end_host = get_nanos();


	printf("%i Elements \n", N);

	printf("  Results: \n");
	printResult("host  ", outHost[N - 1]);
	printResult("device", outDevice[N - 1]);
	printResult("bcao", outDeviceBcao[N - 1]);
	
	printf("  Time: \n");
	printTimeElapsed("host  ", start_host, end_host);
	printTimeElapsed("device", start, end);
	printTimeElapsed("bcao  ", start_bcao, end_bcao);

	printf("\n\n");


	delete[] in;
	delete[] outDevice;
	delete[] outHost;
}

int main()
{
	int TEN_MILLION = 10000000;
	int ONE_MILLION = 1000000;
	int TEN_THOUSAND = 10000;

	int elements[] = {
		TEN_MILLION,
		ONE_MILLION,
		TEN_THOUSAND,
		5000,
		4096,
		2048,
		2000,
		1000,
		100,
		64,
		8,
		5
	};

	int numElements = sizeof(elements) / sizeof(elements[0]);

	for (int i = 0; i < numElements; i++) {
		//test(elements[i]);
	}

	test_blockscan(1024);
	test_blockscan(512);
	test_blockscan(64);
	test_blockscan(8);

	return 0;
}