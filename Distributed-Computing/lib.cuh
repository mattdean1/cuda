void sequential_scan(int* output, int* input, int length);
__global__ void naive_scan(int *g_odata, int *g_idata, int n);
__global__ void prescan(int *g_odata, int *g_idata, int n);
__global__ void prescan_large(int *g_odata, int *g_idata, int n);