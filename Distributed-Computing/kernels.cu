#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include "kernels.cuh"

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5

__device__ int CONFLICT_FREE_OFFSET(int n, bool bcao) {
	if (bcao) {
		return ((n) >> SHARED_MEMORY_BANKS + (n) >> (2 * LOG_MEM_BANKS));
	}
	else {
		return ((n) >> LOG_MEM_BANKS);
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

__global__ void prescan(int *output, int *input, int n, bool bcao) {
	extern __shared__ int temp[];

	int threadID = threadIdx.x;
	int offset = 1;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai, bcao);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi, bcao);
	temp[ai + bankOffsetA] = input[ai];
	temp[bi + bankOffsetB] = input[bi];

	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai, bcao);
			bi += CONFLICT_FREE_OFFSET(bi, bcao);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) { 
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1, bcao)] = 0; // clear the last element
	} 

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai, bcao);
			bi += CONFLICT_FREE_OFFSET(bi, bcao);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[ai] = temp[ai + bankOffsetA];
	output[bi] = temp[bi + bankOffsetB];
}

__global__ void prescan_arbitrary(int *g_odata, int *g_idata, int n, int powerOfTwo)
{
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	if (threadID < n) {
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

	if (threadID < n) {
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

__global__ void add(int *output, int length, int *n) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}

__global__ void add(int *output, int length, int *n1, int *n2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}