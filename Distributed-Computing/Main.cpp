#include <stdlib.h>
#include <stdio.h>

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



int main()
{
	int TEN_MILLION = 10000000;
	int ONE_MILLION = 1000000;
	int TEN_THOUSAND = 10000;
	int FIVE_THOUSAND = 10000;

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
		test(elements[i]);
	}

	return 0;
}