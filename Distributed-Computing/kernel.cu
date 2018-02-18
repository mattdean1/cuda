
#include <stdio.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lib.cuh"

void checkCudaError(char *message, cudaError_t err);
void printArray(int* arr, int length, char* prefix);

int main()
{
	// allocate arrays
    int N = 128;
    int *in, *out;

	// managed memory can be accessed by host and gpu - it is slightly slower than cudaMalloc + cudaMemcpy
	cudaMallocManaged(&in, N * sizeof(int));
	cudaMallocManaged(&out, N * sizeof(int));

	// populate arrays
	for (int i = 0; i < N; i++) {
		in[i] = i + 1;
		out[i] = 0;
	}

	prescan<<<1, N/2, N*sizeof(int)>>>(out, in, N);

	checkCudaError(
		"kernel launch",
		cudaGetLastError()
	);
	checkCudaError(
		"kernel execution",
		cudaDeviceSynchronize()
	);


	printArray(in, N, "input array");
	printArray(out, N, "scanned array");

	cudaFree(in);
	cudaFree(out);
    return 0;
}

void checkCudaError(char *message, cudaError_t err) {
	if (err != cudaSuccess) {
		fprintf(stderr, message);
		fprintf(stderr, ": %s\n", cudaGetErrorString(err));
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
