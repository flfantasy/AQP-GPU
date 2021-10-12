// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR
#define TIMES 100
#define THREADS_PER_BLOCK 32
#define NUM_SM 72
#define WARPS_PER_SM 32
#define THREADS_PER_WARP 32
#define MAX_CONCURRENT_THREADS (NUM_SM * WARPS_PER_SM * THREADS_PER_WARP) 

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
  for(int i = 0; i < MAX_CONCURRENT_THREADS; i++) {
    curand_init(clock_for_rand + i, 0, 0, curand_states + i);
  }
}

// 进行一次BS试验的采样部分
__global__ void create_BS_sample(
    int* bs_lo_orderdate, int* bs_lo_discount, int* bs_lo_quantity, int* bs_lo_extendedprice,
    int* d_lo_orderdate, int* d_lo_discount, int* d_lo_quantity, int* d_lo_extendedprice,
    curandState *curand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < TIMES) {
    for (size_t i = 0; i < LO_LEN; i++)
    {
      int smid = __mysmid();
      int warpid = __mywarpid();
      int laneid = __mylaneid();
      unsigned int rand = curand(curand_states + (smid * 1024 + warpid * 32 + laneid));
      int loidx = rand % LO_LEN;
      int idx = x * LO_LEN + i;
      bs_lo_orderdate[idx] = d_lo_orderdate[loidx];
      bs_lo_discount[idx] = d_lo_discount[loidx];
      bs_lo_quantity[idx] = d_lo_quantity[loidx];
      bs_lo_extendedprice[idx] = d_lo_extendedprice[loidx];
    }
  }
}

// 进行一次BS试验的查询部分，使用crystal
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void queryKernel(int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice,
    unsigned long long* res) {
  // items1表示某一列中由这个thread处理的几行
  // items2表示另一列中由这个thread处理的几行
  // selection_flags是一个bitmap，表示是否通过过滤
  int items1[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  long long sum = 0;

  // 当前tile在整个数组中的offset
  int tile_offset = blockIdx.x * TILE_SIZE;
  // tile的数量，LO_LEN/TILE_SIZE 向上取整
  int num_tiles = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
  // 当前tile内有多少items
  int num_tile_items = TILE_SIZE;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = LO_LEN - tile_offset;
  }

  // lo_orderdate >= 19930101 and lo_orderdate < 19940101
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset, items1, num_tile_items);
  BlockPredGT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items1, 19930000, selection_flags, num_tile_items);
  BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items1, 19940000, selection_flags, num_tile_items);

  // lo_quantity<25
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_quantity + tile_offset, items1, num_tile_items);
  BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items1, 25, selection_flags, num_tile_items);

  // lo_discount>=1 and lo_discount<=3
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_discount + tile_offset, items1, num_tile_items);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items1, 1, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items1, 3, selection_flags, num_tile_items);

  // lo_extendedprice
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_extendedprice + tile_offset, items2, num_tile_items);

  // 计算一个thread的sum
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      if (selection_flags[ITEM])
        sum += items1[ITEM] * items2[ITEM];
  }
  __syncthreads();

  // buffer用于存储每个warp的sum，最多支持32个warp
  static __shared__ long long buffer[32];
  
  // 计算整个block的sum
  unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(res, aggregate);
  }
}

void run(int* h_lo_orderdate, int* h_lo_discount, int* h_lo_quantity, int* h_lo_extendedprice, 
  cub::CachingDeviceAllocator&  g_allocator) {
  // load column data to device memory
  int *d_lo_orderdate = loadToGPU<int>(h_lo_orderdate, LO_LEN, g_allocator);
  int *d_lo_discount = loadToGPU<int>(h_lo_discount, LO_LEN, g_allocator);
  int *d_lo_quantity = loadToGPU<int>(h_lo_quantity, LO_LEN, g_allocator);
  int *d_lo_extendedprice = loadToGPU<int>(h_lo_extendedprice, LO_LEN, g_allocator);

  // BS样本
  int* bs_lo_orderdate = NULL;
  int* bs_lo_discount = NULL;
  int* bs_lo_quantity = NULL;
  int* bs_lo_extendedprice = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&bs_lo_orderdate, TIMES * LO_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&bs_lo_discount, TIMES * LO_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&bs_lo_quantity, TIMES * LO_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&bs_lo_extendedprice, TIMES * LO_LEN * sizeof(int)));
  cudaMemset(bs_lo_orderdate, 0, TIMES * LO_LEN * sizeof(int));
  cudaMemset(bs_lo_discount, 0, TIMES * LO_LEN * sizeof(int));
  cudaMemset(bs_lo_quantity, 0, TIMES * LO_LEN * sizeof(int));
  cudaMemset(bs_lo_extendedprice, 0, TIMES * LO_LEN * sizeof(int));

  // 记录TIMES次bootstarp的sum
  unsigned long long* d_res = NULL;
  unsigned long long* d_bs_res = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_res, sizeof(unsigned long long)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_bs_res, TIMES * sizeof(unsigned long long)));
  cudaMemset(d_bs_res, 0, TIMES * sizeof(unsigned long long));
  unsigned long long h_res;
  unsigned long long h_bs_res[TIMES];

  // cuda注册start、stop事件，用于计时
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float time1 = 0.0f;
  float time2 = 0.0f;
  float unit_time = 0.0f;
  cudaEventRecord(start, 0);
  // 随机数生成器初始化，为了防止冲突需要设置多个。titan的并行度是72个sm * 32个warp/sm * 32个thread/warp
  curandState *curand_state;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&curand_state, MAX_CONCURRENT_THREADS * sizeof(curandState)));
  long clock_for_rand = clock();  //程序运行时钟数
  curandGenKernel<<<1, 1>>>(curand_state, clock_for_rand);    //main road
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&unit_time, start, stop);
  time1 += unit_time;

  cudaEventRecord(start, 0);
  {
    int num_blocks = (TIMES - 1) / THREADS_PER_BLOCK + 1;
    create_BS_sample<<<num_blocks, THREADS_PER_BLOCK>>>(bs_lo_orderdate, bs_lo_discount, bs_lo_quantity, bs_lo_extendedprice, d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, curand_state);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&unit_time, start, stop);
  time1 += unit_time;
  for (int i = 0; i < TIMES; i++) {
    cudaEventRecord(start, 0);
    {
      int tile_items = 128*4;
      int num_blocks = (LO_LEN - 1) / tile_items + 1;
      int offset = i * LO_LEN;
      queryKernel<128,4><<<num_blocks, 128>>>(bs_lo_orderdate + offset, bs_lo_discount + offset, bs_lo_quantity + offset, bs_lo_extendedprice + offset, d_bs_res+i);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&unit_time, start, stop);
    time2 += unit_time;
  }
  cout << "]" << endl;
  cout << "Time Taken(resample): " << time1 << "ms" << endl;

  cudaEventRecord(start, 0);
  CubDebugExit(cudaMemcpy(&h_bs_res, d_bs_res, TIMES * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  sort(h_bs_res, h_bs_res + TIMES);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&unit_time, start, stop);
  time2 += unit_time;

  int tile_items = 128*4;
  int num_blocks = (LO_LEN - 1) / tile_items + 1;
  queryKernel<128,4><<<num_blocks, 128>>>(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, d_res);
  CubDebugExit(cudaMemcpy(&h_res, d_res, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

  int idx1 = TIMES * 0.01;
  int idx2 = TIMES * 0.99;
  cout << h_res << " (" << (double)((long long)h_bs_res[idx1]-(long long)h_res)/h_res << ", " << (double)((long long)h_bs_res[idx2]-h_res)/h_res << ")" << endl; 
  cout << "Time Taken(run query): " << time2 << "ms" << endl;

  CLEANUP(curand_state);
  CLEANUP(bs_lo_orderdate);
  CLEANUP(bs_lo_discount);
  CLEANUP(bs_lo_quantity);
  CLEANUP(bs_lo_extendedprice);
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

  // 注册st、finish时间点，c++的计时工具
	chrono::high_resolution_clock::time_point st, finish;
  st = chrono::high_resolution_clock::now();
  run(h_lo_orderdate, h_lo_discount, h_lo_quantity, h_lo_extendedprice, g_allocator);
  finish = chrono::high_resolution_clock::now();
  chrono::duration<double> diff = finish - st;
  cout << "total time: " << diff.count() * 1000 << "ms" << endl;
  return 0;
}