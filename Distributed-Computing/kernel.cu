
#include <stdio.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lib.cuh"

void checkCudaError(char *message, cudaError_t err);
void printArray(int* arr, int length, char* prefix);
bool isPowerOfTwo(int x);
int nextPowerOfTwo(int x);
void addConstant(int *output, int length, int constant);
void scanSmallArray(int *output, int *input, int length);
void scanLargeArray(int *output, int *input, int length);
void scanSmallArbitraryArray(int *output, int *input, int length);
void scanLargeArbitraryArray(int *output, int *input, int length);

int main()
{
	// allocate arrays
    const int N = 2046;

	int in[N];
	int out[N] = { 0 };

	// populate arrays
	for (int i = 0; i < N; i++) {
		in[i] = i + 1;
		out[i] = 0;
	}

	scanLargeArbitraryArray(out, in, N);

	printArray(in, N, "input array");
	printArray(out, N, "scanned array");

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
	const int numElementsPerBlock = 512;
	const int blocks = length / numElementsPerBlock;
	const int threadsPerBlock = numElementsPerBlock / 2;
	const int arraySize = length * sizeof(int);

	int *d_out, *d_in, *d_sums, *d_sums2, *d_incr;

	cudaMalloc((void **)&d_out, arraySize);
	cudaMalloc((void **)&d_in, arraySize);
	cudaMalloc((void **)&d_sums, blocks * sizeof(int));
	cudaMalloc((void **)&d_sums2, blocks * sizeof(int));
	cudaMalloc((void **)&d_incr, blocks * sizeof(int));

	cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);


	prescan_large<<<blocks, threadsPerBlock, arraySize>>>(d_out, d_in, numElementsPerBlock, d_sums);
	checkCudaError(
		"kernel launch",
		cudaGetLastError()
	);
	checkCudaError(
		"kernel execution",
		cudaDeviceSynchronize()
	);

	prescan<<<1, blocks / 2, blocks * sizeof(int)>>>(d_incr, d_sums, blocks);
	checkCudaError(
		"kernel launch",
		cudaGetLastError()
	);
	checkCudaError(
		"kernel execution",
		cudaDeviceSynchronize()
	);


	add<<<blocks, numElementsPerBlock>>>(d_out, d_incr, numElementsPerBlock);
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
	cudaFree(d_sums);
}

void scanLargeArbitraryArray(int *output, int *input, int length) {
	bool isp2 = isPowerOfTwo(length);
	if (isp2) {
		scanLargeArray(output, input, length);
	}
	else {
		int prevPower = nextPowerOfTwo(length) / 2;
		int remaining = length - prevPower;

		scanLargeArray(output, input, prevPower);
		int last = output[prevPower - 1];

		scanSmallArbitraryArray(&(output[prevPower]), &(input[prevPower]), remaining);
		addConstant(&(output[prevPower]), remaining, last + input[prevPower - 1]);
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