
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lib.cuh"


void printArray(int* arr, int length, const char* prefix);
bool isPowerOfTwo(int x);
int nextPowerOfTwo(int x);
void addConstant(int *output, int length, int constant);
void scanSmallArray(int *output, int *input, int length);
void scanLargeArray(int *output, int *input, int length);
void scanSmallArbitraryArray(int *output, int *input, int length);
void scanLargeArbitraryArray(int *output, int *input, int length);

void _checkCudaError(const char *message, cudaError_t err, const char *caller) {
	if (err != cudaSuccess) {
		fprintf(stderr, "Error in: %s\n", caller);
		fprintf(stderr, message);
		fprintf(stderr, ": %s\n", cudaGetErrorString(err));
		exit(0);
	}
}

#define checkCudaError(o, l) _checkCudaError(o, l, __func__)

int THREADS_PER_BLOCK = 512;

int main()
{
    const int N = 1000000;

	int *in = new int[N];
	int *outDevice = new int[N]();
	int *outHost = new int[N]();

	for (int i = 0; i < N; i++) {
		in[i] = rand()%10;
	}

	scanLargeArbitraryArray(outDevice, in, N);
	sequential_scan(outHost, in, N);

	printf("device: %i\n", outDevice[N - 1]);
	printf("host: %i\n", outHost[N - 1]);

	//printArray(in, N, "input array");
	//printArray(out, N, "scanned array");

	delete[] in;
	delete[] outDevice;
	delete[] outHost;

    return 0;
}

void scanSmallArray(int *output, int *input, int length) {
	const int arraySize = length * sizeof(int);
	int *d_out, *d_in;
	cudaMalloc((void **)&d_out, arraySize);
	cudaMalloc((void **)&d_in, arraySize);

	cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);

	prescan<<<1, length / 2, arraySize>>>(d_out, d_in, length);
	checkCudaError(
		"kernel launch",
		cudaGetLastError()
	);
	checkCudaError(
		"kernel execution",
		cudaDeviceSynchronize()
	);

	cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);
}

void scanSmallArbitraryArray(int *output, int *input, int length) {
	const int arraySize = length * sizeof(int);
	int *d_out, *d_in;
	cudaMalloc((void **)&d_out, arraySize);
	cudaMalloc((void **)&d_in, arraySize);

	cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);

	int powerOfTwo = nextPowerOfTwo(length);

	prescan_arbitrary<<<1, length / 2, powerOfTwo * sizeof(int)>>>(d_out, d_in, length, powerOfTwo);
	checkCudaError(
		"kernel launch",
		cudaGetLastError()
	);
	checkCudaError(
		"kernel execution",
		cudaDeviceSynchronize()
	);

	cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);
}

void scanLargeArray(int *output, int *input, int length) {
	const int numElementsPerBlock = THREADS_PER_BLOCK * 2;
	const int blocks = length / numElementsPerBlock;
	const int arraySize = length * sizeof(int);

	int *d_out, *d_in, *d_sums, *d_sums2, *d_incr;

	cudaMalloc((void **)&d_out, arraySize);
	cudaMalloc((void **)&d_in, arraySize);
	cudaMalloc((void **)&d_sums, blocks * sizeof(int));
	cudaMalloc((void **)&d_sums2, blocks * sizeof(int));
	cudaMalloc((void **)&d_incr, blocks * sizeof(int));

	cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);

	const int sharedArraySize = numElementsPerBlock * sizeof(int);
	prescan_large<<<blocks, THREADS_PER_BLOCK, sharedArraySize>>>(d_out, d_in, numElementsPerBlock, d_sums);
	checkCudaError(
		"prescan_large kernel launch",
		cudaGetLastError()
	);
	checkCudaError(
		"prescan_large kernel execution",
		cudaDeviceSynchronize()
	);

	int powerOfTwo = nextPowerOfTwo(blocks);
	prescan_arbitrary<<<1, (blocks + 1) / 2, powerOfTwo * sizeof(int)>>>(d_incr, d_sums, blocks, powerOfTwo);
	checkCudaError(
		"prescan kernel launch",
		cudaGetLastError()
	);
	checkCudaError(
		"prescan kernel execution",
		cudaDeviceSynchronize()
	);


	add<<<blocks, numElementsPerBlock>>>(d_out, d_incr, numElementsPerBlock);
	checkCudaError(
		"add kernel launch",
		cudaGetLastError()
	);
	checkCudaError(
		"add kernel execution",
		cudaDeviceSynchronize()
	);

	cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);
	cudaFree(d_sums);
}

void scanLargeArbitraryArray(int *output, int *input, int length) {
	int remainder = length % (THREADS_PER_BLOCK * 2);
	if (remainder == 0) {
		scanLargeArray(output, input, length);
	}
	else {
		int lengthMultiple = length - remainder;

		scanLargeArray(output, input, lengthMultiple);
		int lastElem = output[lengthMultiple - 1];

		scanSmallArbitraryArray(&(output[lengthMultiple]), &(input[lengthMultiple]), remainder);
		addConstant(&(output[lengthMultiple]), remainder, lastElem + input[lengthMultiple - 1]);
	}
}

void addConstant(int *output, int length, int constant) {
	int *d_out, *d_add;
	int arraySize = length * sizeof(int);

	cudaMalloc((void **)&d_out, arraySize);
	cudaMalloc((void **)&d_add, sizeof(int));

	cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_add, &constant, sizeof(int), cudaMemcpyHostToDevice);

	add << <1, length >> >(d_out, d_add, length);
	checkCudaError(
		"kernel launch",
		cudaGetLastError()
	);
	checkCudaError(
		"kernel execution",
		cudaDeviceSynchronize()
	);

	cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
}



void printArray(int* arr, int length, const char* prefix) {
	printf(prefix);
	printf(": {");
	for (int i = 0; i < length; i++) {
		printf(" %i", arr[i]);
	}
	printf(" }\n");
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