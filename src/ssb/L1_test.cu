# include <stdio.h>
# include <stdint.h>
# include <unistd.h>
# include <iostream>
# include "cuda_runtime.h"

using namespace std;

//compile nvcc *.cu -o test

__global__ void global_latency (unsigned int * array, int array_length, int iterations,
    unsigned int *index, unsigned int * duration, unsigned int * duration1) {
	unsigned int start_time, mid_time, end_time;
	unsigned int j = 100;
	// __shared__ unsigned int s_index[9*1024];
	__shared__ unsigned int s_index;
	int k;
	// for(k=0; k<9*1024; k++){
	// 	s_index[k] = 0;
	// }
	for (k = 0; k < iterations; k++) {
		if(k>=0){
			start_time = clock();
			j = array[j];
			mid_time = clock();
			// s_index[k]= j;
			s_index = j;
			end_time = clock();
			duration[k] = mid_time - start_time;
			duration1[k] = end_time - mid_time;
		}
		else j = array[j];
	}
	// for(k=0; k<9*1024; k++){
	// 	index[k]= s_index[k];
	// }
}

void parametric_measure_global(int N, int iterations, int stride) {
	cudaDeviceReset();
	int i;
	unsigned int * h_a = (unsigned int *)malloc(sizeof(unsigned int) * (N+2));
	unsigned int * d_a;
	cudaMalloc((void **) &d_a, sizeof(unsigned int) * (N+2));

	for (i = 0; i < N; i++) {
		h_a[i] = (i+stride)%N;
	}
	h_a[N] = 0;
	h_a[N+1] = 0;
    cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);

	unsigned int *h_index = (unsigned int *)malloc(sizeof(unsigned int)*iterations);
	unsigned int *h_duration = (unsigned int *)malloc(sizeof(unsigned int)*iterations);
	unsigned int *h_duration1 = (unsigned int *)malloc(sizeof(unsigned int)*iterations);
	unsigned int *d_index;
	cudaMalloc( (void **) &d_index, sizeof(unsigned int)*iterations);
	unsigned int *d_duration;
	cudaMalloc ((void **) &d_duration, sizeof(unsigned int)*iterations);
	unsigned int *d_duration1;
	cudaMalloc ((void **) &d_duration1, sizeof(unsigned int)*iterations);

	cudaThreadSynchronize ();
	cudaFuncSetAttribute(global_latency, cudaFuncAttributePreferredSharedMemoryCarveout, 50);	// 100为shared mem占用L1cache的百分比
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp,0);
	cout << deviceProp.sharedMemPerBlock << endl;
	cout << deviceProp.sharedMemPerBlockOptin << endl;
	cout << deviceProp.sharedMemPerMultiprocessor << endl;
	
	global_latency <<<1, 1>>>(d_a, N, iterations, d_index, d_duration, d_duration1);

{	/* copy results from GPU to CPU */
	cudaThreadSynchronize ();
    cudaMemcpy((void *)h_index, (void *)d_index, sizeof(unsigned int)*iterations, cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)h_duration, (void *)d_duration, sizeof(unsigned int)*iterations, cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)h_duration1, (void *)d_duration1, sizeof(unsigned int)*iterations, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize ();}

{	int tavg = 0;
	int tavg1 = 0;
	for(i=0;i<iterations;i++) {
		printf("%d\t %d\n", h_index[i], h_duration[i]);
		usleep(100);
		tavg += h_duration[i];
		tavg1 += h_duration1[i];
	}
	tavg /= iterations;
	tavg1 /= iterations;
	printf("%d, %d\n", tavg, tavg1);}

{	cudaFree(d_a);
	cudaFree(d_index);
	cudaFree(d_duration);
    free(h_a);
    free(h_index);
	free(h_duration);
	cudaDeviceReset();	}
}

void measure_global() {
	int N, iterations, stride;
	//stride in element
	iterations = 67;
	//1. overflow cache with 1 element. stride=1, N=4097
	//2. overflow cache with cache lines. stride=32, N_min=16*256, N_max=24*256	
	// stride = 128/sizeof(unsigned int);
	stride = 128;
	for (N = 1024; N <= 1024; N+=stride) {
		printf("\n=====%10.4f KB array, warm TLB, read 512 element====\n", sizeof(unsigned int)*(float)N/1024);
		printf("Stride = %d element, %d byte\n", stride, stride * sizeof(unsigned int));
		parametric_measure_global(N, iterations, stride);
		printf("===============================================\n\n");
	}
}

int main(){
	cudaSetDevice(1);
	measure_global();
	cudaDeviceReset();
	return 0;
}





