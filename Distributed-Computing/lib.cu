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

__global__ void naive_scan(int *g_odata, int *g_idata, int n)
{
	extern __shared__ int temp[]; // allocated on invocation
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

__global__ void prescan_arbitrary(int *g_odata, int *g_idata, int n, int powerOfTwo)
{
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	if (threadID <= n) {
		temp[2 * threadID] = g_idata[2 * threadID]; // load input into shared memory
		temp[2 * threadID + 1] = g_idata[2 * threadID + 1];
	}
	else {
		temp[2 * threadID] = 0;
		temp[2 * threadID + 1] = 0;
	}
	

	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
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

	if (threadID == 0) { temp[powerOfTwo - 1] = 0; } // clear the last element

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
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

	if (threadID <= n) {
		g_odata[2 * threadID] = temp[2 * threadID]; // write results to device memory
		g_odata[2 * threadID + 1] = temp[2 * threadID + 1];
	}
}

__global__ void prescan_large(int *output, int *input, int n, int *sums) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;
	
	extern __shared__ int temp[];
	temp[2 * threadID] = input[blockOffset + (2 * threadID)];
	temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];

	int offset = 1;
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
	__syncthreads();

	
	if (threadID == 0) { 
		sums[blockID] = temp[n - 1];
		temp[n - 1] = 0;
	} 
	
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

	output[blockOffset + (2 * threadID)] = temp[2 * threadID];
	output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
}

__global__ void add(int *output, int* addit, int n) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;

	output[blockOffset + threadID] += addit[blockID];
}