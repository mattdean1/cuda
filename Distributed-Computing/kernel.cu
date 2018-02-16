
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "lib.cuh"

void scanArray(int *output, int *input, int size);
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
	naive_scan1<<<1, N, N>>>(out, in, N);

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


// Helper function for using CUDA to add vectors in parallel.
//void scanArray(int *output, int *input, int size)
//{
//	int *device_output, *device_output;
//	cudaError_t err;
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//	err = cudaMalloc((void**)&device_output, size * sizeof(int));
//	if (err != cudaSuccess) {
//		fprintf(stderr, "error allocating memory: %s\n", cudaGetErrorString(err));
//		exit(0);
//	}
//
//	err = cudaMalloc((void**)&device_input, size * sizeof(int));
//	if (err != cudaSuccess) {
//		fprintf(stderr, "error allocating memory: %s\n", cudaGetErrorString(err));
//		exit(0);
//	}
//
//    // Copy input vectors from host memory to GPU buffers.
//    err = cudaMemcpy(device_input, input, size * sizeof(int), cudaMemcpyHostToDevice);
//	if (err != cudaSuccess) {
//		fprintf(stderr, "error copying h to d: %s\n", cudaGetErrorString(err));
//		exit(0);
//	}
//
//    // Launch a kernel on the GPU with one thread for each element.
//    naive_scan1<<<1, size>>>(device_output, device_input, size);
//
//    // Check for any errors launching the kernel
//    err = cudaGetLastError();
//    if (err != cudaSuccess) {
//        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
//		exit(0);
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the execution.
//	err = cudaDeviceSynchronize();
//	if (err != cudaSuccess) {
//		fprintf(stderr, "sync failed: %s\n", cudaGetErrorString(err));
//		exit(0);
//	}
//
//    // Copy output vector from GPU buffer to host memory.
//	err = cudaMemcpy(output, device_output, size * sizeof(int), cudaMemcpyDeviceToHost);
//	if (err != cudaSuccess) {
//		fprintf(stderr, "copying d to h failed: %s\n", cudaGetErrorString(err));
//		exit(0);
//	}
//
//    //cudaFree(device_input);
//    //cudaFree(device_output);
//}
