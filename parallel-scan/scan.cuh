long sequential_scan(int* output, int* input, int length);
float blockscan(int *output, int *input, int length, bool bcao);
float scan(int *output, int *input, int length, bool bcao);

void scanLargeDeviceArray(int *output, int *input, int length, bool bcao);
void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool bcao);
void scanLargeEvenDeviceArray(int *output, int *input, int length, bool bcao);