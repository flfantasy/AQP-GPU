// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include "cub/test/test_util.h"

#include "crystal/crystal.cuh"

#include "gpu_utils.h"
#include "ssb_utils.h"

using namespace std;

/**
 * Globals, constants and typedefs
 */
bool                    g_verbose = false;  // Whether to display input/output to console
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory, 用于给变量分配设备内存

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void QueryKernel(int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice,
    int lo_num_entries, unsigned long long* revenue) {
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];

  long long sum = 0;

  // 当前tile在整个数组中的offset
  int tile_offset = blockIdx.x * TILE_SIZE;
  // tile的数量
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

  // 计算一个tile（128 * 4）的sum
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      if (selection_flags[ITEM])
        sum += items[ITEM] * items2[ITEM];
  }

  __syncthreads();

  // 共享变量buffer存储中间结果
  static __shared__ long long buffer[32];
  
  unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(revenue, aggregate);
  }
}

// 此函数主要是记录核函数的时间，传入四列数组（的显存地址）
float runQuery(int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice, 
    int lo_num_entries, cub::CachingDeviceAllocator&  g_allocator) {
  // 注册start、stop事件
  SETUP_TIMING();

  float time_query;
  chrono::high_resolution_clock::time_point st, finish;
  st = chrono::high_resolution_clock::now();

  cudaEventRecord(start, 0);

  unsigned long long* d_sum = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum, sizeof(long long)));

  cudaMemset(d_sum, 0, sizeof(long long));

  // Run
  int tile_items = 128*4;
  int num_blocks = (lo_num_entries + tile_items - 1)/tile_items;
  QueryKernel<128,4><<<num_blocks, 128>>>(lo_orderdate, 
          lo_discount, lo_quantity, lo_extendedprice, lo_num_entries, d_sum);

  // 计算时间
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_query, start,stop);

  unsigned long long revenue;
  CubDebugExit(cudaMemcpy(&revenue, d_sum, sizeof(long long), cudaMemcpyDeviceToHost));

  finish = chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = finish - st;

  // diff = cpu数据传给gpu时间 + gpu执行时间
  cout << "Revenue: " << revenue << endl;
  cout << "Time Taken Total: " << diff.count() * 1000 << endl;

  CLEANUP(d_sum);
  // time_query = diff + 结果传回cpu时间
  return time_query;
}

/**
 * Main
 */
int main(int argc, char** argv)
{
  int num_trials          = 3;

  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("t", num_trials);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
      printf("%s "
          "[--t=<num trials>] "
          "[--v] "
          "\n", argv[0]);
      exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  // load column data to host memory
  int *h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
  int *h_lo_discount = loadColumn<int>("lo_discount", LO_LEN);
  int *h_lo_quantity = loadColumn<int>("lo_quantity", LO_LEN);
  int *h_lo_extendedprice = loadColumn<int>("lo_extendedprice", LO_LEN);
  int *h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int *h_d_year = loadColumn<int>("d_year", D_LEN);

  cout << "** LOADED DATA **" << endl;
  cout << "LO_LEN " << LO_LEN << endl;

  // load column data to device memory
  int *d_lo_orderdate = loadToGPU<int>(h_lo_orderdate, LO_LEN, g_allocator);
  int *d_lo_discount = loadToGPU<int>(h_lo_discount, LO_LEN, g_allocator);
  int *d_lo_quantity = loadToGPU<int>(h_lo_quantity, LO_LEN, g_allocator);
  int *d_lo_extendedprice = loadToGPU<int>(h_lo_extendedprice, LO_LEN, g_allocator);
  int *d_d_datekey = loadToGPU<int>(h_d_datekey, D_LEN, g_allocator);
  int *d_d_year = loadToGPU<int>(h_d_year, D_LEN, g_allocator);

  cout << "** LOADED DATA TO GPU **" << endl;

  for (int t = 0; t < num_trials; t++) {
    // 查询执行时间
    float time_query;
    time_query = runQuery(d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN, g_allocator);
    cout<< "{"
        << "\"query\":11" 
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  return 0;
}

