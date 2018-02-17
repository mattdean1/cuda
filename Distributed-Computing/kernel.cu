
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lib.cuh"

void checkCudaError(cudaError_t err);

int main()
{
	// allocate arrays
    const int N = 5;
    int *in, *out;

	// managed memory can be accessed by host and gpu - it is slightly slower than cudaMalloc + cudaMemcpy
	cudaMallocManaged(&in, N * sizeof(int));
	cudaMallocManaged(&out, N * sizeof(int));

	for (int i = 0; i < N; i++) {
		in[i] = i + 1;
		out[i] = 0;
	}

	// launch the kernel
	naive_scan<<<1, N, N>>>(out, in, N);

	checkCudaError(
		// check that the kernel was launched ok
		cudaGetLastError()
	);
	checkCudaError(
		// check the kernel executed ok
		cudaDeviceSynchronize()
	);

    printf("scan{1,2,3,4,5} = {%d,%d,%d,%d,%d}\n", out[0], out[1], out[2], out[3], out[4]);

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
