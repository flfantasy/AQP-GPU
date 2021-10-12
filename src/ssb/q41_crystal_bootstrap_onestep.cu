// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR
#define TIMES 128
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
#include <unordered_map>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cub/util_allocator.cuh>
#include "cub/test/test_util.h"

#include "crystal/crystal.cuh"

#include "gpu_utils.h"
#include "ssb_utils.h"

using namespace std;

struct pair_hash
{
    template<class T1, class T2>
    size_t operator() (const pair<T1, T2>& p) const
    {
        auto h1 = hash<T1>{}(p.first);
        auto h2 = hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

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

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_c(int* filter_col, int *dim_key, int* dim_val, int *hash_table) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (C_LEN + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = C_LEN - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, items2, selection_flags, 
      hash_table, C_SLOT_LEN, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s(int *filter_col, int *dim_key, int *hash_table) {
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (S_LEN + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = S_LEN - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
  BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, 
      hash_table, S_SLOT_LEN, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_p(int *filter_col, int *dim_key, int *hash_table) {
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (P_LEN + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = P_LEN - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 0, selection_flags, num_tile_items);
  BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
  BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, 
      hash_table, P_SLOT_LEN, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_d(int *dim_key, int *dim_val, int *hash_table) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (D_LEN + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = D_LEN - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, items2, selection_flags,
      hash_table, D_SLOT_LEN, D_VAL_MIN, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe(int* lo_orderdate, int* lo_custkey, int* lo_suppkey, int* lo_partkey, int* lo_revenue, int* lo_supplycost,
    int* ht_c, int* ht_s, int* ht_p, int* ht_d, int* res) {
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int c_nation[ITEMS_PER_THREAD];
  int d_year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = LO_LEN - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset, items, num_tile_items);
  BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, ht_s, S_SLOT_LEN, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_custkey + tile_offset, items, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, c_nation, selection_flags,
      ht_c, C_SLOT_LEN, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_partkey + tile_offset, items, num_tile_items);
  BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, ht_p, P_SLOT_LEN, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset, items, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, d_year, selection_flags,
      ht_d, D_SLOT_LEN, D_VAL_MIN, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_revenue + tile_offset, revenue, num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_supplycost + tile_offset, items, num_tile_items);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items) {
      if (selection_flags[ITEM]) {
        int hash = (c_nation[ITEM] * 7 +  (d_year[ITEM] - 1992)) % ((1998-1992+1) * 25);
        res[hash * 4] = d_year[ITEM];
        res[hash * 4 + 1] = c_nation[ITEM];
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]), (long long)(revenue[ITEM] - items[ITEM]));
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void createBSSampleAndProbe(
    int* lo_orderdate, int* lo_custkey, int* lo_suppkey, int* lo_partkey, int* lo_revenue, int* lo_supplycost,
    int* ht_c, int* ht_s, int* ht_p, int* ht_d,
    int* res, curandState* curand_states) {
  // Load a tile striped across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int c_nation[ITEMS_PER_THREAD];
  int d_year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = LO_LEN - tile_offset;
  }

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < LO_LEN) {
    int smid = __mysmid();
    int warpid = __mywarpid();
    int laneid = __mylaneid();
    unsigned int rand = curand(curand_states + (smid * 1024 + warpid * 32 + laneid));
    int loidx = rand % LO_LEN; 
    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);
    items[0] = lo_suppkey[loidx];
    BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, ht_s, S_SLOT_LEN, num_tile_items);
    items[0] = lo_custkey[loidx];
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, c_nation, selection_flags,
        ht_c, C_SLOT_LEN, num_tile_items);
    items[0] = lo_partkey[loidx];
    BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, ht_p, P_SLOT_LEN, num_tile_items);
    items[0] = lo_orderdate[loidx];
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, d_year, selection_flags,
        ht_d, D_SLOT_LEN, D_VAL_MIN, num_tile_items);
    revenue[0] = lo_revenue[loidx];
    items[0] = lo_supplycost[loidx];
    if (selection_flags[0]) {
      int hash = (c_nation[0] * 7 +  (d_year[0] - 1992)) % ((1998-1992+1) * 25);
      res[hash * 4] = d_year[0];
      res[hash * 4 + 1] = c_nation[0];
      atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]), (long long)(revenue[0] - items[0]));
    }
  }
}

void createBSSampleAndRunQuery(
    int* lo_orderdate, int* lo_custkey, int* lo_suppkey, int* lo_partkey, int* lo_revenue, int* lo_supplycost,
    int* c_custkey, int* c_nation, int* c_region,
    int* s_suppkey, int* s_region,
    int* p_partkey, int* p_mfgr,
    int* d_datekey, int* d_year,
    unordered_map<pair<int, int>, vector<long long>, pair_hash>* bs_umap) {

  // 存储单次查询结果
  int* d_res;
  int res_size = ((1998-1992+1) * 25);
  int res_array_size = res_size * 4;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int)));
  CubDebugExit(cudaMemset(d_res, 0, res_array_size * sizeof(int)));
  int* h_res = new int[res_array_size];

  // 三张hash表
  int *d_ht_c, *d_ht_s, *d_ht_p, *d_ht_d;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_c, 2 * C_SLOT_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * S_SLOT_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_p, 2 * P_SLOT_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * D_SLOT_LEN * sizeof(int)));
  CubDebugExit(cudaMemset(d_ht_c, 0, 2 * C_SLOT_LEN * sizeof(int)));
  CubDebugExit(cudaMemset(d_ht_s, 0, 2 * S_SLOT_LEN * sizeof(int)));
  CubDebugExit(cudaMemset(d_ht_p, 0, 2 * P_SLOT_LEN * sizeof(int)));
  CubDebugExit(cudaMemset(d_ht_d, 0, 2 * D_SLOT_LEN * sizeof(int)));

  // 随机数生成器初始化，为了防止冲突需要设置多个。titan的并行度是72个sm * 32个warp/sm * 32个thread/warp
  curandState *curand_state;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&curand_state, MAX_CONCURRENT_THREADS * sizeof(curandState)));
  long clock_for_rand = clock();  //程序运行时钟数
  curandGenKernel<<<1, 1>>>(curand_state, clock_for_rand);    //main road

  cout << "TIMES: " << TIMES << "[";
  for (int i = 0; i < TIMES; i++) {
    if (i % 10 == 0) cout << i;
    cout << "=" << flush;
    {
      int items_per_tile = 128*4;
      int num_blocks = (C_LEN - 1) / items_per_tile + 1;
      build_hashtable_c<128,4><<<num_blocks, 128>>>(c_region, c_custkey, c_nation, d_ht_c);
      num_blocks = (S_LEN - 1) / items_per_tile + 1;
      build_hashtable_s<128,4><<<num_blocks, 128>>>(s_region, s_suppkey, d_ht_s);
      num_blocks = (P_LEN - 1) / items_per_tile + 1;
      build_hashtable_p<128,4><<<num_blocks, 128>>>(p_mfgr, p_partkey, d_ht_p);
      num_blocks = (D_LEN - 1) / items_per_tile + 1;
      build_hashtable_d<128,4><<<num_blocks, 128>>>(d_datekey, d_year, d_ht_d);
      num_blocks = (LO_LEN - 1) / THREADS_PER_BLOCK + 1;
      CubDebugExit(cudaMemset(d_res, 0, res_array_size * sizeof(int)));
      createBSSampleAndProbe<THREADS_PER_BLOCK,1><<<num_blocks, THREADS_PER_BLOCK>>>(
          lo_orderdate, lo_custkey, lo_suppkey, lo_partkey, lo_revenue, lo_supplycost,
          d_ht_c, d_ht_s, d_ht_p, d_ht_d,
          d_res, curand_state);
      CubDebugExit(cudaMemcpy(h_res, d_res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < res_size; i++) {
        if (h_res[4*i] != 0) {
          int d_year = h_res[4*i];
          int c_nation = h_res[4*i + 1];
          long long sum = reinterpret_cast<long long*>(&h_res[4*i + 2])[0];
          pair<int, int> p(d_year, c_nation);
          (*bs_umap)[p].push_back(sum);
        }
      }
    }
  }
  cout << "]" << endl;
  for (auto& entry : *bs_umap) {
    vector<long long>& vec = entry.second;
    sort(vec.begin(), vec.end());
  }
  CLEANUP(d_ht_c);
  CLEANUP(d_ht_s);
  CLEANUP(d_ht_p);
  CLEANUP(d_ht_d);
  CLEANUP(curand_state);
}

void runQuery(
    int* lo_orderdate, int* lo_custkey, int* lo_suppkey, int* lo_partkey, int* lo_revenue, int* lo_supplycost,
    int* c_custkey, int* c_nation, int* c_region,
    int* s_suppkey, int* s_region,
    int* p_partkey, int* p_mfgr,
    int* d_datekey, int* d_year,
    unordered_map<pair<int, int>, long long, pair_hash>* umap) {

  // 存储单次查询结果
  int* d_res;
  int res_size = ((1998-1992+1) * 25);
  int res_array_size = res_size * 4;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int)));
  CubDebugExit(cudaMemset(d_res, 0, res_array_size * sizeof(int)));
  int* h_res = new int[res_array_size];

  // 三张hash表
  int *d_ht_c, *d_ht_s, *d_ht_p, *d_ht_d;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_c, 2 * C_SLOT_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * S_SLOT_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_p, 2 * P_SLOT_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * D_SLOT_LEN * sizeof(int)));
  CubDebugExit(cudaMemset(d_ht_c, 0, 2 * C_SLOT_LEN * sizeof(int)));
  CubDebugExit(cudaMemset(d_ht_s, 0, 2 * S_SLOT_LEN * sizeof(int)));
  CubDebugExit(cudaMemset(d_ht_p, 0, 2 * P_SLOT_LEN * sizeof(int)));
  CubDebugExit(cudaMemset(d_ht_d, 0, 2 * D_SLOT_LEN * sizeof(int)));

  int items_per_tile = 128*4;
  int num_blocks = (C_LEN - 1) / items_per_tile + 1;
  build_hashtable_c<128,4><<<num_blocks, 128>>>(c_region, c_custkey, c_nation, d_ht_c);
  num_blocks = (S_LEN - 1) / items_per_tile + 1;
  build_hashtable_s<128,4><<<num_blocks, 128>>>(s_region, s_suppkey, d_ht_s);
  num_blocks = (P_LEN - 1) / items_per_tile + 1;
  build_hashtable_p<128,4><<<num_blocks, 128>>>(p_mfgr, p_partkey, d_ht_p);
  num_blocks = (D_LEN - 1) / items_per_tile + 1;
  build_hashtable_d<128,4><<<num_blocks, 128>>>(d_datekey, d_year, d_ht_d);
  num_blocks = (LO_LEN - 1) / items_per_tile + 1;
  probe<128,4><<<num_blocks, 128>>>(
      lo_orderdate, lo_custkey, lo_suppkey, lo_partkey, lo_revenue, lo_supplycost,
      d_ht_c, d_ht_s, d_ht_p, d_ht_d,
      d_res);
  CubDebugExit(cudaMemcpy(h_res, d_res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < res_size; i++) {
    if (h_res[4*i] != 0) {
      int d_year = h_res[4*i];
      int c_nation = h_res[4*i + 1];
      long long sum = reinterpret_cast<long long*>(&h_res[4*i + 2])[0];
      pair<int, int> p(d_year, c_nation);
      (*umap)[p] = sum;
    }
  }
  CLEANUP(d_ht_c);
  CLEANUP(d_ht_s);
  CLEANUP(d_ht_p);
  CLEANUP(d_ht_d);
}

void run(
    int* h_lo_orderdate, int* h_lo_custkey, int* h_lo_suppkey, int* h_lo_partkey, int* h_lo_revenue, int* h_lo_supplycost,
    int* h_c_custkey, int* h_c_nation, int* h_c_region,
    int* h_s_suppkey, int* h_s_region,
    int* h_p_partkey, int* h_p_mfgr,
    int* h_d_datekey, int* h_d_year,
    cub::CachingDeviceAllocator&  g_allocator) {
  // load column data to device memory
  int *d_lo_orderdate = loadToGPU<int>(h_lo_orderdate, LO_LEN, g_allocator);
  int *d_lo_custkey = loadToGPU<int>(h_lo_custkey, LO_LEN, g_allocator);
  int *d_lo_suppkey = loadToGPU<int>(h_lo_suppkey, LO_LEN, g_allocator);
  int *d_lo_partkey = loadToGPU<int>(h_lo_partkey, LO_LEN, g_allocator);
  int *d_lo_revenue = loadToGPU<int>(h_lo_revenue, LO_LEN, g_allocator);
  int *d_lo_supplycost = loadToGPU<int>(h_lo_supplycost, LO_LEN, g_allocator);
  int *d_c_custkey = loadToGPU<int>(h_c_custkey, C_LEN, g_allocator);
  int *d_c_nation = loadToGPU<int>(h_c_nation, C_LEN, g_allocator);
  int *d_c_region  = loadToGPU<int>(h_c_region, C_LEN, g_allocator);
  int *d_s_suppkey = loadToGPU<int>(h_s_suppkey, S_LEN, g_allocator);
  int *d_s_region = loadToGPU<int>(h_s_region, S_LEN, g_allocator);
  int *d_p_partkey = loadToGPU<int>(h_p_partkey, P_LEN, g_allocator);
  int *d_p_mfgr = loadToGPU<int>(h_p_mfgr, P_LEN, g_allocator);
  int *d_d_datekey = loadToGPU<int>(h_d_datekey, D_LEN, g_allocator);
  int *d_d_year = loadToGPU<int>(h_d_year, D_LEN, g_allocator);


	// 原始样本和bs样本的query值
  unordered_map<pair<int, int>, long long, pair_hash> umap;
  unordered_map<pair<int, int>, vector<long long>, pair_hash> bs_umap;

  // cuda注册start、stop事件，用于计时
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float time = 0.0f;
  cudaEventRecord(start, 0);
  createBSSampleAndRunQuery(
      d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_partkey, d_lo_revenue, d_lo_supplycost,
      d_c_custkey, d_c_nation, d_c_region,
      d_s_suppkey, d_s_region,
      d_p_partkey, d_p_mfgr,
      d_d_datekey, d_d_year,
      &bs_umap);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cout << "Time Taken(resample + run query): " << time << "ms" << endl;

  // 计算原始样本的query值
  runQuery(
      d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_partkey, d_lo_revenue, d_lo_supplycost,
      d_c_custkey, d_c_nation, d_c_region,
      d_s_suppkey, d_s_region,
      d_p_partkey, d_p_mfgr,
      d_d_datekey, d_d_year,
      &umap);

  for (auto& entry1 : umap) {
    auto p1 = entry1.first;
    long long sum = entry1.second;
    if(bs_umap.find(p1) == bs_umap.end()){
        cout << p1.first << "\t" << p1.second << "\t" << sum << "(0,0)" << endl;
      continue;
    }
    vector<long long> bs_sum = bs_umap[p1];
    int length = bs_sum.size();
    int idx1 = length * 0.01;
    int idx2 = length * 0.99;
    cout << p1.first << "\t" << p1.second << "\t";
    cout << sum << "(" << (double)(bs_sum[idx1]-sum)/sum << "," << (double)(bs_sum[idx2]-sum)/sum << ")" << endl;
  }
  cout << "Res Count: " << umap.size() << endl;

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
  int *h_lo_custkey = loadColumn<int>("lo_custkey", LO_LEN);
  int *h_lo_suppkey = loadColumn<int>("lo_suppkey", LO_LEN);
  int *h_lo_partkey = loadColumn<int>("lo_partkey", LO_LEN);
  int *h_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);
  int *h_lo_supplycost = loadColumn<int>("lo_supplycost", LO_LEN);
  int *h_c_custkey = loadColumn<int>("c_custkey", C_LEN);
  int *h_c_nation = loadColumn<int>("c_nation", C_LEN);
  int *h_c_region = loadColumn<int>("c_region", C_LEN);
  int *h_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
  int *h_s_region = loadColumn<int>("s_region", S_LEN);
  int *h_p_partkey = loadColumn<int>("p_partkey", P_LEN);
  int *h_p_mfgr = loadColumn<int>("p_mfgr", P_LEN);
  int *h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int *h_d_year = loadColumn<int>("d_year", D_LEN);


  // 注册st、finish时间点，c++的计时工具
	chrono::high_resolution_clock::time_point st, finish;
  st = chrono::high_resolution_clock::now();
  run(
      h_lo_orderdate, h_lo_custkey, h_lo_suppkey, h_lo_partkey, h_lo_revenue, h_lo_supplycost,
      h_c_custkey, h_c_nation, h_c_region,
      h_s_suppkey, h_s_region,
      h_p_partkey, h_p_mfgr,
      h_d_datekey, h_d_year,
      g_allocator);
  finish = chrono::high_resolution_clock::now();
  chrono::duration<double> diff = finish - st;
  cout << "total time: " << diff.count() * 1000 << "ms" << endl;
  return 0;
}