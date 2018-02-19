#include "cuda_runtime.h"

void _checkCudaError(const char *message, cudaError_t err, const char *caller);
void printResult(const char* prefix, int number);
void printArray(int* arr, int length, const char* prefix);

bool isPowerOfTwo(int x);
int nextPowerOfTwo(int x);

void sequential_scan(int* output, int* input, int length);
