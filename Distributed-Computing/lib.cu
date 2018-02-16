#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>


#include "lib.cuh"


void sequential_scan(int* output, int* input, int length)
{
	output[0] = 0; // since this is a prescan, not a scan
	for (int j = 1; j < length; ++j)
	{
		output[j] = input[j - 1] + output[j - 1];
	}
}

__global__ void naive_scan1(int *g_odata, int *g_idata, int n)
{
	__shared__ int temp[5]; // allocated on invocation
	int k = threadIdx.x;
	// load input into shared memory.
	// This is exclusive scan, so shift right by one and set first elt to 0
	temp[k] = (k > 0) ? g_idata[k - 1] : 0;
	// sync threads so all elements in temp are allocated
	__syncthreads();

	for (int d = 1; d < n; d *= 2)
	{
		if (k >= d)
			temp[k] += temp[k - d];
		__syncthreads();
	}
	g_odata[k] = temp[k]; // write output
}

__global__ void naive_scan(int *g_odata, int *g_idata, int n)
{
	extern __shared__ int temp[]; // allocated on invocation

	int threadID = threadIdx.x;
	int pout = 0, pin = 1;

	// load input into shared memory.
	// This is exclusive scan, so shift right by one and set first elt to 0
	temp[pout*n + threadID] = (threadID > 0) ? g_idata[threadID - 1] : 0;
	__syncthreads();

	for (int offset = 1; offset < n; offset *= 2)
	{
		pout = 1 - pout; // swap double buffer indices
		pin = 1 - pout;

		if (threadID >= offset)
			temp[pout*n + threadID] += temp[pin*n + threadID - offset];
		else
			temp[pout*n + threadID] = temp[pin*n + threadID];

		 __syncthreads();
	}
	g_odata[threadID] = temp[pout*n + threadID]; // write output
}

__global__ void prescan(int *g_odata, int *g_idata, int n)
{
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;
	int offset = 1;
	temp[2 * threadID] = g_idata[2 * threadID]; // load input into shared memory
	temp[2 * threadID + 1] = g_idata[2 * threadID + 1];
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (threadID == 0) { temp[n - 1] = 0; } // clear the last element
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	g_odata[2 * threadID] = temp[2 * threadID]; // write results to device memory
	g_odata[2 * threadID + 1] = temp[2 * threadID + 1];
}