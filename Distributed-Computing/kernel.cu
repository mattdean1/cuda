
#include <stdio.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lib.cuh"

void checkCudaError(cudaError_t err);
void printArray(int* arr, int length, char* prefix);

int main()
{
	// allocate arrays
    const int N = 8;
    int *in, *out;

	// managed memory can be accessed by host and gpu - it is slightly slower than cudaMalloc + cudaMemcpy
	cudaMallocManaged(&in, N * sizeof(int));
	cudaMallocManaged(&out, N * sizeof(int));

	for (int i = 0; i < N; i++) {
		in[i] = i + 1;
		out[i] = 0;
	}

	// launch the kernel
	prescan<<<1, N, 2*N>>>(out, in, N);

	checkCudaError(
		// check that the kernel was launched ok
		cudaGetLastError()
	);
	checkCudaError(
		// check the kernel executed ok
		cudaDeviceSynchronize()
	);


	printArray(in, N, "input array");
	printArray(out, N, "scanned array");

	cudaFree(in);
	cudaFree(out);
    return 0;
}

void checkCudaError(cudaError_t err) {
	if (err != cudaSuccess) {
		fprintf(stderr, "cuda error: %s\n", cudaGetErrorString(err));
		exit(0);
	}
}

void printArray(int* arr, int length, char* prefix) {
	char string[50] = "";
	strcat(strcat(string, prefix), ": {");

	printf(string);
	for (int i = 0; i < length; i++) {
		printf(" %i", arr[i]);
	}
	printf(" }\n");
}
