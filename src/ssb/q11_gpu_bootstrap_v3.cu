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

__global__ void create_BS_sample(int* bs_sample,
    int lo_num_entries, curandState *curand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < lo_num_entries) {
    int smid = __mysmid();
    int warpid = __mywarpid();
    int laneid = __mylaneid();
    unsigned int rand = curand(curand_states + (smid * 1024 + warpid * 32 + laneid));
    atomicAdd(reinterpret_cast<unsigned int*>(&bs_sample[rand % lo_num_entries]), 1); 
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void QueryKernel(int* bs_sample,
    int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice,
    int lo_num_entries, unsigned long long* d_sum) {
  // items表示某一列中由这个thread处理的几行
  // items表示另一列中由这个thread处理的几行
  // selection_flags是一个bitmap，表示是否通过过滤
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int items3[ITEMS_PER_THREAD];

  long long sum = 0;

  // 当前tile在整个数组中的offset
  int tile_offset = blockIdx.x * TILE_SIZE;
  // tile的数量，lo_num_entries/TILE_SIZE 向上取整
  int num_tiles = (lo_num_entries + TILE_SIZE - 1) / TILE_SIZE;
  // 当前tile内有多少items
  int num_tile_items = TILE_SIZE;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = lo_num_entries - tile_offset;
  }

  // lo_orderdate >= 19930101 and lo_orderdate < 19940101
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset, items, num_tile_items);
  BlockPredGT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19930000, selection_flags, num_tile_items);
  BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 19940000, selection_flags, num_tile_items);

  // lo_quantity<25
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_quantity + tile_offset, items, num_tile_items);
  BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 25, selection_flags, num_tile_items);

  // lo_discount>=1 and lo_discount<=3
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_discount + tile_offset, items, num_tile_items);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 3, selection_flags, num_tile_items);

  // lo_extendedprice
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_extendedprice + tile_offset, items2, num_tile_items);
  // bs_sample
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(bs_sample + tile_offset, items3, num_tile_items);

  // 计算一个thread的sum
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      if (selection_flags[ITEM])
        sum += items[ITEM] * items2[ITEM] * items3[ITEM];
  }
  __syncthreads();

  // buffer用于存储每个warp的sum，最多支持32个warp
  static __shared__ long long buffer[32];
  
  // 计算整个block的sum
  unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(d_sum, aggregate);
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

  // BS样本
  int* bs_sample = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&bs_sample, LO_LEN * sizeof(int)));
  cudaMemset(bs_sample, 0, LO_LEN * sizeof(int));

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

  for (int i = 0; i < TIMES; i++) {
    int num_blocks = (lo_num_entries - 1) / threads_per_block + 1;
    create_BS_sample<<<num_blocks, threads_per_block>>>(bs_sample, lo_num_entries, curand_state);    //main road
    int tile_items = 128*4;
    num_blocks = (lo_num_entries - 1) / tile_items + 1;
    QueryKernel<128,4><<<num_blocks, 128>>>(bs_sample, d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, lo_num_entries, d_sum+i);
    cudaMemset(bs_sample, 0, LO_LEN * sizeof(int));
  }
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