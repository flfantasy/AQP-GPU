#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>

#define blockSize 1024

#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

#define MAX_SM 72

unsigned long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}


__device__ unsigned long long count = 0;
__device__ unsigned int blk_ids[MAX_SM] = {0};

__global__ void rng_init(unsigned long long seed, curandState * states) {
  const size_t Idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, Idx, 0, &states[Idx]);
}

__global__ void kernel(curandState * states, int length) {
  const size_t Idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < length; i++){
    const float x = curand_uniform(&states[Idx]);
    const float y = curand_uniform(&states[Idx]);
    if (sqrtf(x*x+y*y)<1.0)
      atomicAdd(&count, 1ULL);}
}

static __device__ __inline__ int __mysmid(){
  int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;}

__device__ int get_my_resident_thread_id(int sm_blk_id){
  int my_sm = __mysmid();
  if (sm_blk_id != 0) {
    printf("smid: %d, block_id: %d\n", my_sm, sm_blk_id);
  }
  return my_sm * sm_blk_id + threadIdx.x;
}

__device__ int get_block_id(){
  int my_sm = __mysmid();
  int my_block_id = -1;
  bool done = false;
  int i = 0;
  while ((!done)&&(i<32)){
    unsigned int block_flag = 1<<i;
    if ((atomicOr(blk_ids+my_sm, block_flag)&block_flag) == 0){my_block_id = i; done = true;}
    i++;}
  return my_block_id;
}

__device__ void release_block_id(int block_id){
  unsigned int block_mask = ~(1<<block_id);
  int my_sm = __mysmid();
  atomicAnd(blk_ids+my_sm, block_mask);
}

__global__ void kernel2(curandState * states, int length) {

  __shared__ volatile int my_block_id;
  if (!threadIdx.x) my_block_id = get_block_id();
  __syncthreads();
  if (my_block_id != 0) {printf("%d\n",my_block_id);}
  const size_t Idx = get_my_resident_thread_id(my_block_id);
  for (int i = 0; i < length; i++){
    const float x = curand_uniform(&states[Idx]);
    const float y = curand_uniform(&states[Idx]);
    if (sqrtf(x*x+y*y)<1.0)
      atomicAdd(&count, 1ULL);}
  __syncthreads();
  if (!threadIdx.x) release_block_id(my_block_id);
  __syncthreads();
}



int main(int argc, char *argv[]) {
  int gridSize = 10;
  if (argc > 1) gridSize = atoi(argv[1]);
  curandState * states;
  assert(cudaMalloc(&states, gridSize*gridSize*blockSize*sizeof(curandState)) == cudaSuccess);
  unsigned long long hcount;
  //warm - up
  rng_init<<<gridSize*gridSize,blockSize>>>(1234ULL, states);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  //method 1: 1 curand state per point
  std::cout << "Method 1 init blocks: " << gridSize*gridSize << std::endl;
  unsigned long long dtime = dtime_usec(0);
  rng_init<<<gridSize*gridSize,blockSize>>>(1234ULL, states);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  unsigned long long initt = dtime_usec(dtime);
  kernel<<<gridSize*gridSize,blockSize>>>(states, 1);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  dtime = dtime_usec(dtime) - initt;
  cudaMemcpyFromSymbol(&hcount, count, sizeof(unsigned long long));
  std::cout << "method 1 elapsed time: " << dtime/(float)USECPSEC << " init time: " << initt/(float)USECPSEC << " pi: " << 4.0f*hcount/(float)(gridSize*gridSize*blockSize) << std::endl;
  hcount = 0;
  cudaMemcpyToSymbol(count, &hcount, sizeof(unsigned long long));
  //method 2: 1 curand state per gridSize points
  std::cout << "Method 2 init blocks: " << gridSize << std::endl;
  dtime = dtime_usec(0);
  rng_init<<<gridSize,blockSize>>>(1234ULL, states);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  initt = dtime_usec(dtime);
  kernel<<<gridSize,blockSize>>>(states, gridSize);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  dtime = dtime_usec(dtime) - initt;
  cudaMemcpyFromSymbol(&hcount, count, sizeof(unsigned long long));
  std::cout << "method 2 elapsed time: " << dtime/(float)USECPSEC << " init time: " << initt/(float)USECPSEC << " pi: " << 4.0f*hcount/(float)(gridSize*gridSize*blockSize) << std::endl;
  hcount = 0;
  cudaMemcpyToSymbol(count, &hcount, sizeof(unsigned long long));
  //method 3: 1 curand state per resident thread
  // compute the maximum number of state entries needed
  int num_sms;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  int max_sm_threads;
  cudaDeviceGetAttribute(&max_sm_threads, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
  int max_blocks = max_sm_threads/blockSize;
  int total_state = max_blocks*num_sms*blockSize;
  int rgridSize = (total_state + blockSize-1)/blockSize;
  std::cout << "Method 3 sms: " << num_sms << " init blocks: " << rgridSize << std::endl;
  // run test
  dtime = dtime_usec(0);
  rng_init<<<rgridSize,blockSize>>>(1234ULL, states);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  initt = dtime_usec(dtime);
  kernel2<<<gridSize,blockSize>>>(states, gridSize);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  dtime = dtime_usec(dtime) - initt;
  cudaMemcpyFromSymbol(&hcount, count, sizeof(unsigned long long));
  std::cout << "method 3 elapsed time: " << dtime/(float)USECPSEC << " init time: " << initt/(float)USECPSEC << " pi: " << 4.0f*hcount/(float)(gridSize*gridSize*blockSize) << std::endl;
  hcount = 0;
  cudaMemcpyToSymbol(count, &hcount, sizeof(unsigned long long));
  return 0;
}