/*一次kernel调用内计算全部的bs样本的sum*/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR
#define TIMES 128
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

#include "crystal/crystal.cuh"

#include "gpu_utils.h"
#include "ssb_utils.h"

using namespace std;

// Caching allocator for device memory, 用于给变量分配设备内存
cub::CachingDeviceAllocator  g_allocator(true);  

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

__global__ void kernel(int* d_lo_orderdate, int* d_lo_discount, int* d_lo_quantity, int* d_lo_extendedprice,
    int lo_num_entries, unsigned long long* sum, curandState *curand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < lo_num_entries * TIMES) {
    int smid = __mysmid();
    int warpid = __mywarpid();
    int laneid = __mylaneid();
    unsigned int randLO = curand(curand_states + (smid * 1024 + warpid * 32 + laneid));
    int sumidx = x % TIMES;
    int loidx = randLO % lo_num_entries;
    if (d_lo_orderdate[loidx] >= 19930101 && d_lo_orderdate[loidx] < 19940101 && d_lo_quantity[loidx] < 25 && d_lo_discount[loidx] >= 1 && d_lo_discount[loidx] <= 3){
      atomicAdd(&sum[sumidx], (unsigned long long)(d_lo_discount[loidx] * d_lo_extendedprice[loidx]));
    }
  }
}

// bootstrap的方法1：每次采样都atomicadd，sum数组传回CPU再排序
void run(int* h_lo_orderdate, int* h_lo_discount, int* h_lo_quantity, int* h_lo_extendedprice, 
  int lo_num_entries, cub::CachingDeviceAllocator&  g_allocator, int threads_per_block) {

  float time_query;
  // cuda注册start、stop事件，用于计时
  cudaEvent_t start, stop1, stop2, stop3, stop4;
{  cudaEventCreate(&start);
  cudaEventCreate(&stop1);
  cudaEventCreate(&stop2);
  cudaEventCreate(&stop3);
  cudaEventCreate(&stop4);
  cudaEventRecord(start, 0);}

  // load column data to device memory
  int *d_lo_orderdate = loadToGPU<int>(h_lo_orderdate, LO_LEN, g_allocator);
  int *d_lo_discount = loadToGPU<int>(h_lo_discount, LO_LEN, g_allocator);
  int *d_lo_quantity = loadToGPU<int>(h_lo_quantity, LO_LEN, g_allocator);
  int *d_lo_extendedprice = loadToGPU<int>(h_lo_extendedprice, LO_LEN, g_allocator);
{  cudaEventRecord(stop1, 0);
  cudaEventSynchronize(stop1);
  cudaEventElapsedTime(&time_query, start, stop1);
  cout << "H2D时间:" << time_query << "ms" << endl;}

  // TIMES次bootstarp的sum
  unsigned long long* d_sum = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum, TIMES * sizeof(unsigned long long)));
  cudaMemset(d_sum, 0, TIMES * sizeof(unsigned long long));

  // 随机数生成器初始化，为了防止冲突需要设置多个。titan的并行度是72个sm * 32个warp/sm * 32个thread/warp
  curandState *curand_state;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&curand_state, MOST_CONCURRENT_THREADS * sizeof(curandState)));
  long clock_for_rand = clock();  //程序运行时钟数
  curandGenKernel<<<1, 1>>>(curand_state, clock_for_rand);    //main road
{  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  cudaEventElapsedTime(&time_query, stop1, stop2);
  cout << "随机数生成器初始化:" << time_query << "ms" << endl;}

  int num_blocks = (lo_num_entries * TIMES - 1) / threads_per_block + 1;
  kernel<<<num_blocks, threads_per_block>>>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, lo_num_entries, d_sum, curand_state);    //main road
{  cudaEventRecord(stop3, 0);
  cudaEventSynchronize(stop3);
  cudaEventElapsedTime(&time_query, stop2, stop3);
  cout << "GPU采样时间:" << time_query << "ms" << endl;}

{  unsigned long long h_sum[TIMES];
  CubDebugExit(cudaMemcpy(&h_sum, d_sum, TIMES * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  sort(h_sum, h_sum + TIMES);
  cout << h_sum[1] << "," << h_sum[126] << endl;

  cudaEventRecord(stop4, 0);
  cudaEventSynchronize(stop4);
  cudaEventElapsedTime(&time_query, stop3, stop4);
  cout << "D2H并排序时间:" << time_query << "ms" << endl;
  long long sum = 446268068091;
  cout << sum << " (" << (double)((long long)h_sum[1]-sum)/sum << ", " << (double)((long long)h_sum[TIMES-2]-sum)/sum << ")" << endl; 
  CLEANUP(d_sum);
  CLEANUP(curand_state);}
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
  int *h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
  int *h_lo_discount = loadColumn<int>("lo_discount", LO_LEN);
  int *h_lo_quantity = loadColumn<int>("lo_quantity", LO_LEN);
  int *h_lo_extendedprice = loadColumn<int>("lo_extendedprice", LO_LEN);
  // 计算原始样本sum
  long long sum = 0;
  for (int i = 0; i < LO_LEN; i++) {
    if (h_lo_orderdate[i] >= 19930101 && h_lo_orderdate[i] < 19940101 && h_lo_quantity[i] < 25 && h_lo_discount[i] >= 1 && h_lo_discount[i] <= 3){
      sum += (unsigned long long)(h_lo_discount[i] * h_lo_extendedprice[i]);
    }
  }
  cout << "true sum: " << sum << endl;

  // 注册st、finish时间点，c++的计时工具
	chrono::high_resolution_clock::time_point st, finish;
  int threads_per_block = 32;
  st = chrono::high_resolution_clock::now();
  run(h_lo_orderdate, h_lo_discount, h_lo_quantity, h_lo_extendedprice, LO_LEN, g_allocator, threads_per_block);  //main road
  finish = chrono::high_resolution_clock::now();
  chrono::duration<double> diff = finish - st;
  cout << "总时间: " << diff.count() * 1000 << "ms" << endl;

  return 0;
}