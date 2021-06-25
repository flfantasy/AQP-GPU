// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR
#define TIMES 100
#define NUM_SM 72
#define WARPS_PER_SM 32
#define THREADS_PER_WARP 32
#define MOST_CONCURRENT_THREADS (NUM_SM * WARPS_PER_SM * THREADS_PER_WARP) 

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <time.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cub/util_allocator.cuh>
#include "cub/test/test_util.h"

#include "gpu_utils.h"
#include "ssb_utils.h"

using namespace std;

cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory, 用于给变量分配设备内存

static __device__ __inline__ uint32_t __mysmid(){
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

static __device__ __inline__ uint32_t __mywarpid(){
  uint32_t warpid;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;
}

static __device__ __inline__ uint32_t __mylaneid(){    
  uint32_t laneid;    
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));    
  return laneid;
}

__global__ void  curandGenKernel(curandState *curand_states,long clock_for_rand) {
  for(int i = 0; i < MOST_CONCURRENT_THREADS; i++) {
    curand_init(clock_for_rand + i, 0, 0, curand_states + i);
  }
}

__global__ void kernel(int* lo_revenue, int lo_num_entries, unsigned long long* sum,
    curandState *curand_states, unsigned int* d_duration1, unsigned int* d_duration2) {
  // Start timing
  uint32_t start = 0, stop1 = 0, stop2 = 0;
  asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < lo_num_entries * TIMES) {
    int smid = __mysmid();
    int warpid = __mywarpid();
    int laneid = __mylaneid();
    unsigned int rand = curand(curand_states + (smid * 1024 + warpid * 32 + laneid));
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop1) :: "memory");
    int idx = x % TIMES;
    int loidx = rand % lo_num_entries;
    atomicAdd(&sum[idx], (unsigned long long)lo_revenue[loidx]);
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop2) :: "memory");
  }
  d_duration1[x] = stop1 - start;
  d_duration2[x] = stop2 - stop1;
}

// bootstrap的方法1：每次采样都atomicadd，sum数组传回CPU再排序
void run(int* h_lo_revenue, int lo_num_entries, cub::CachingDeviceAllocator&  g_allocator,
  int threads_per_block, unsigned long long* h_low_bound, unsigned long long* h_upper_bound) {
  int num_blocks = (lo_num_entries * TIMES - 1) / threads_per_block + 1;

  // TIMES次bootstarp的sum
  unsigned long long* d_sum1 = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum1, TIMES * sizeof(unsigned long long)));
  cudaMemset(d_sum1, 0, TIMES * sizeof(unsigned long long));
  unsigned int* d_duration1 = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_duration1, num_blocks * threads_per_block * sizeof(unsigned int)));
  cudaMemset(d_duration1, 0, TIMES * sizeof(unsigned int));
  unsigned int* d_duration2 = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_duration2, num_blocks * threads_per_block * sizeof(unsigned int)));
  cudaMemset(d_duration2, 0, TIMES * sizeof(unsigned int));
  int* d_lo_revenue;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_lo_revenue, sizeof(int) * lo_num_entries));
  CubDebugExit(cudaMemcpy(d_lo_revenue, h_lo_revenue, sizeof(int) * lo_num_entries, cudaMemcpyHostToDevice));
  // 随机数生成器初始化，为了防止冲突需要设置多个。titan的并行度是72个sm * 32个warp/sm * 32个thread/warp
  curandState *curand_state;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&curand_state, MOST_CONCURRENT_THREADS * sizeof(curandState)));
  long clock_for_rand = clock();  //程序运行时钟数
  curandGenKernel<<<1, 1>>>(curand_state, clock_for_rand);    //main road

  kernel<<<num_blocks, threads_per_block>>>(d_lo_revenue, lo_num_entries, d_sum1, curand_state, d_duration1, d_duration2);    //main road

  unsigned int h_duration1[num_blocks * threads_per_block];
  CubDebugExit(cudaMemcpy(&h_duration1, d_duration1, num_blocks * threads_per_block * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  unsigned int h_duration2[num_blocks * threads_per_block];
  CubDebugExit(cudaMemcpy(&h_duration2, d_duration2, num_blocks * threads_per_block * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  for(int i = 0; i < num_blocks * threads_per_block; i++){
    cout << h_duration1[i] << " ";
  }
  cout << endl;
  for(int i = 0; i < num_blocks * threads_per_block; i++){
    cout << h_duration2[i] << " ";
  }
  cout << endl;
  unsigned long long h_sum1[TIMES];
  CubDebugExit(cudaMemcpy(&h_sum1, d_sum1, TIMES * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  sort(h_sum1, h_sum1 + TIMES);
  cout << h_sum1[1] << "," << h_sum1[98] << endl;
  *h_low_bound = h_sum1[1];
  *h_upper_bound = h_sum1[98];

  CLEANUP(d_sum1);
  CLEANUP(curand_state);
  CLEANUP(d_lo_revenue);
}

/**
 * Main
 */
int main(int argc, char** argv){
  // Initialize command line
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  // load column data to host memory
  int h_lo_revenue[100];
  for (int i = 0; i < 100; i++) {
    h_lo_revenue[i] = i;
  }
  // 计算原始样本sum
  long long sum = 0;
  for (int i = 0; i < 100; i++) {
    sum += h_lo_revenue[i];
  }

  // 置信区间结果
  unsigned long long low_bound = 0;
  unsigned long long upper_bound = 0;

  // 注册st、finish时间点，c++的计时工具
	chrono::high_resolution_clock::time_point st, finish;
  int threads_per_block = 32;
  st = chrono::high_resolution_clock::now();
  run(h_lo_revenue, 100, g_allocator, threads_per_block, &low_bound, &upper_bound);  //main road
  finish = chrono::high_resolution_clock::now();
  chrono::duration<double> diff = finish - st;
  cout << "总时间: " << diff.count() * 1000 << "ms" << endl;
  cout << sum << "(" << (long)low_bound - sum << "," << upper_bound - sum << ")" << endl;
  cout << sum << "(" << (double)((long)low_bound - sum)/sum << "," << (double)(upper_bound - sum)/sum << ")" << endl;

	return 0;
} 