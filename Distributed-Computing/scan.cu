
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernels.cuh"
#include "helpers.h"
#include "scan.cuh"

#define checkCudaError(o, l) _checkCudaError(o, l, __func__)

int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

long blockscan(int *output, int *input, int length, bool bcao){
	int *d_out, *d_in;
	const int arraySize = length * sizeof(int);

	cudaMalloc((void **)&d_out, arraySize);
	cudaMalloc((void **)&d_in, arraySize);
	cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);

	// start timer
	long start = get_nanos();
	int powerOfTwo = nextPowerOfTwo(length);

	prescan_arbitrary<<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >>>(d_out, d_in, length, powerOfTwo, bcao);
	checkCudaError(
		"kernel launch",
		cudaGetLastError()
	);
	checkCudaError(
		"kernel execution",
		cudaDeviceSynchronize()
	);

	// end timer
	cudaDeviceSynchronize();
	long end = get_nanos();

	cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);

	return end - start;
}


long scan(int *output, int *input, int length, bool bcao) {
	int *d_out, *d_in;
	const int arraySize = length * sizeof(int);

	cudaMalloc((void **)&d_out, arraySize);
	cudaMalloc((void **)&d_in, arraySize);
	cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);

	// start timer
	long start = get_nanos();

	if (length > ELEMENTS_PER_BLOCK) {
		scanLargeDeviceArray(d_out, d_in, length, bcao);
	}
	else {
		scanSmallDeviceArray(d_out, d_in, length, bcao);
	}

	// end timer
	cudaDeviceSynchronize();
	long end = get_nanos();

	cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);

	return end - start;
}

void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
	int remainder = length % (ELEMENTS_PER_BLOCK);
	if (remainder == 0) {
		scanLargeEvenDeviceArray(d_out, d_in, length, bcao);
	}
	else {
		// perform a large scan on a compatible multiple of elements
		int lengthMultiple = length - remainder;
		scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, bcao);

		// scan the remaining elements and add the last element of the large scan to this
		scanSmallDeviceArray(&(d_out[lengthMultiple]), &(d_in[lengthMultiple]), remainder, bcao);

		add<<<1, remainder>>>(&(d_out[lengthMultiple]), remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
		checkCudaError(
			"kernel launch",
			cudaGetLastError()
		);
		checkCudaError(
			"kernel execution",
			cudaDeviceSynchronize()
		);
	}
}

void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
	int powerOfTwo = nextPowerOfTwo(length);

	prescan_arbitrary<<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int)>>>(d_out, d_in, length, powerOfTwo, bcao);
	checkCudaError(
		"kernel launch",
		cudaGetLastError()
	);
	checkCudaError(
		"kernel execution",
		cudaDeviceSynchronize()
	);
}

void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

	int *d_sums, *d_incr;
	cudaMalloc((void **)&d_sums, blocks * sizeof(int));
	cudaMalloc((void **)&d_incr, blocks * sizeof(int));

	prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums, bcao);
	checkCudaError(
		"prescan_large kernel launch",
		cudaGetLastError()
	);
	checkCudaError(
		"prescan_large kernel execution",
		cudaDeviceSynchronize()
	);

	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
		// perform a large scan on the sums arr
		scanLargeDeviceArray(d_incr, d_sums, blocks, bcao);
	}
	else {
		// only need one block to scan sums arr so can use small scan
		scanSmallDeviceArray(d_incr, d_sums, blocks, bcao);
	}

	add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);
	checkCudaError(
		"add kernel launch",
		cudaGetLastError()
	);
	checkCudaError(
		"add kernel execution",
		cudaDeviceSynchronize()
	);

	cudaFree(d_sums);
	cudaFree(d_incr);
}