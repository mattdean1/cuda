#include <stdio.h>
#include <process.h>
#include <time.h>

#include "cuda_runtime.h"

void _checkCudaError(const char *message, cudaError_t err, const char *caller) {
	if (err != cudaSuccess) {
		fprintf(stderr, "Error in: %s\n", caller);
		fprintf(stderr, message);
		fprintf(stderr, ": %s\n", cudaGetErrorString(err));
		exit(0);
	}
}

void printResult(const char* prefix, int result, long time) {
	printf("    ");
	printf(prefix);
	printf(" : %i in %ld ms \n", result, time / 1000);
}


bool isPowerOfTwo(int x) {
	return x && !(x & (x - 1));
}

int nextPowerOfTwo(int x) {
	int power = 1;
	while (power < x) {
		power *= 2;
	}
	return power;
}


void sequential_scan(int* output, int* input, int length)
{
	output[0] = 0; // since this is a prescan, not a scan
	for (int j = 1; j < length; ++j)
	{
		output[j] = input[j - 1] + output[j - 1];
	}
}

long get_nanos() {
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}
