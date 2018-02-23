__global__ void prescan_arbitrary(int *g_odata, int *g_idata, int n, int powerOfTwo);
__global__ void prescan_arbitrary_unoptimized(int *g_odata, int *g_idata, int n, int powerOfTwo);

__global__ void prescan_large(int *g_odata, int *g_idata, int n, int* sums);
__global__ void prescan_large_unoptimized(int *output, int *input, int n, int *sums);

__global__ void add(int *output, int length, int *n1);
__global__ void add(int *output, int length, int *n1, int *n2);