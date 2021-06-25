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

struct tuple_hash
{
  template<class T1, class T2, class T3>
  size_t operator() (const tuple<T1, T2, T3>& t) const
  {
      auto h1 = hash<T1>{}(get<0>(t));
      auto h2 = hash<T2>{}(get<1>(t));
      auto h3 = hash<T3>{}(get<2>(t));
      return h1 ^ h2 ^ h3;
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

// 进行一次BS试验的采样部分
__global__ void create_BS_sample(
    int* bs_lo_orderdate, int* bs_lo_custkey, int* bs_lo_suppkey, int* bs_lo_revenue,
    int* d_lo_orderdate, int* d_lo_custkey, int* d_lo_suppkey, int* d_lo_revenue,
    curandState *curand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < LO_LEN) {
    int smid = __mysmid();
    int warpid = __mywarpid();
    int laneid = __mylaneid();
    unsigned int rand = curand(curand_states + (smid * 1024 + warpid * 32 + laneid));
    int loidx = rand % LO_LEN;
    bs_lo_orderdate[x] = d_lo_orderdate[loidx];
    bs_lo_custkey[x] = d_lo_custkey[loidx];
    bs_lo_suppkey[x] = d_lo_suppkey[loidx];
    bs_lo_revenue[x] = d_lo_revenue[loidx];    
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s(int *dim_key, int *dim_val, int *hash_table) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (S_LEN + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = S_LEN - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, 231, selection_flags, num_tile_items);
  BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, 235, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, items2, selection_flags, 
      hash_table, S_SLOT_LEN, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_c(int *dim_key, int* dim_val, int *hash_table) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (C_LEN + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = C_LEN - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, 231, selection_flags, num_tile_items);
  BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, 235, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, items2, selection_flags, 
      hash_table, C_SLOT_LEN, num_tile_items);
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

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items, num_tile_items);
  BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1992, selection_flags, num_tile_items);
  BlockPredLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1997, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, items, selection_flags, 
      hash_table, D_SLOT_LEN, D_VAL_MIN, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe(int* lo_orderdate, int* lo_custkey, int* lo_suppkey, int* lo_revenue,
    int* ht_c, int* ht_s, int* ht_d,
    int* res) {
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int c_city[ITEMS_PER_THREAD];
  int s_city[ITEMS_PER_THREAD];
  int d_year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = LO_LEN - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_custkey + tile_offset, items, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, c_city, selection_flags,
      ht_c, C_SLOT_LEN, num_tile_items);
      
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset, items, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, s_city, selection_flags,
      ht_s, S_SLOT_LEN, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_orderdate + tile_offset, items, num_tile_items);
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, d_year, selection_flags,
      ht_d, D_SLOT_LEN, D_VAL_MIN, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_revenue + tile_offset, revenue, num_tile_items);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) {
      if (selection_flags[ITEM]) {
        int hash = (s_city[ITEM] * 250 * 7  + c_city[ITEM] * 7 +  (d_year[ITEM] - 1992)) % ((1998-1992+1) * 250 * 250);
        res[hash * 4] = c_city[ITEM];
        res[hash * 4 + 1] = s_city[ITEM];
        res[hash * 4 + 2] = d_year[ITEM];
        atomicAdd(&res[hash * 4 + 3], revenue[ITEM]);
      }
    }
  }
}

// 进行一次BS试验的查询部分，使用crystal
void queryKernel(
    int* lo_orderdate, int* lo_custkey, int* lo_suppkey, int* lo_revenue,
    int* c_custkey, int* c_city,
    int* s_suppkey, int* s_city,
    int* d_datekey, int* d_year,
    int* ht_c, int* ht_s, int* ht_d, int* res) {

  int items_per_tile = 128*4;
  int num_blocks = (C_LEN - 1) / items_per_tile + 1;
  build_hashtable_c<128,4><<<num_blocks, 128>>>(c_custkey, c_city, ht_c);
  num_blocks = (S_LEN - 1) / items_per_tile + 1;
  build_hashtable_s<128,4><<<num_blocks, 128>>>(s_suppkey, s_city, ht_s);
  num_blocks = (D_LEN - 1) / items_per_tile + 1;
  build_hashtable_d<128,4><<<num_blocks, 128>>>(d_datekey, d_year, ht_d);
  num_blocks = (LO_LEN - 1) / items_per_tile + 1;
  probe<128,4><<<num_blocks, 128>>>(
      lo_orderdate, lo_custkey, lo_suppkey, lo_revenue,
      ht_c, ht_s, ht_d,
      res);
}

void run(
    int* h_lo_orderdate, int* h_lo_custkey, int* h_lo_suppkey, int* h_lo_revenue,
    int* h_c_custkey, int* h_c_city,
    int* h_s_suppkey, int* h_s_city,
    int* h_d_datekey, int* h_d_year,
  cub::CachingDeviceAllocator&  g_allocator) {
  // load column data to device memory
  int *d_lo_orderdate = loadToGPU<int>(h_lo_orderdate, LO_LEN, g_allocator);
  int *d_lo_custkey = loadToGPU<int>(h_lo_custkey, LO_LEN, g_allocator);
  int *d_lo_suppkey = loadToGPU<int>(h_lo_suppkey, LO_LEN, g_allocator);
  int *d_lo_revenue = loadToGPU<int>(h_lo_revenue, LO_LEN, g_allocator);
  int *d_c_custkey = loadToGPU<int>(h_c_custkey, C_LEN, g_allocator);
  int *d_c_city  = loadToGPU<int>(h_c_city, C_LEN, g_allocator);
  int *d_s_suppkey = loadToGPU<int>(h_s_suppkey, S_LEN, g_allocator);
  int *d_s_city = loadToGPU<int>(h_s_city, S_LEN, g_allocator);
  int *d_d_datekey = loadToGPU<int>(h_d_datekey, D_LEN, g_allocator);
  int *d_d_year = loadToGPU<int>(h_d_year, D_LEN, g_allocator);


  // BS样本
  int* bs_lo_orderdate = NULL;
  int* bs_lo_custkey = NULL;
  int* bs_lo_suppkey = NULL;
  int* bs_lo_revenue = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&bs_lo_orderdate, LO_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&bs_lo_custkey, LO_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&bs_lo_suppkey, LO_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&bs_lo_revenue, LO_LEN * sizeof(int)));
  cudaMemset(bs_lo_orderdate, 0, LO_LEN * sizeof(int));
  cudaMemset(bs_lo_custkey, 0, LO_LEN * sizeof(int));
  cudaMemset(bs_lo_suppkey, 0, LO_LEN * sizeof(int));
  cudaMemset(bs_lo_revenue, 0, LO_LEN * sizeof(int));

	// 原始样本和bs样本的query值
  unordered_map<tuple<int, int, int>, int, tuple_hash> umap;
  unordered_map<tuple<int, int, int>, vector<int>, tuple_hash> bs_umap;
  // 三张hash表
  int *d_ht_c, *d_ht_s, *d_ht_d;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_c, 2 * C_SLOT_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * S_SLOT_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * D_SLOT_LEN * sizeof(int)));
  CubDebugExit(cudaMemset(d_ht_c, 0, 2 * C_SLOT_LEN * sizeof(int)));
  CubDebugExit(cudaMemset(d_ht_s, 0, 2 * S_SLOT_LEN * sizeof(int)));
  CubDebugExit(cudaMemset(d_ht_d, 0, 2 * D_SLOT_LEN * sizeof(int)));
  // 存储单次查询结果
  int* d_res;
  int res_size = ((1998-1992+1) * 250 * 250);
  int res_array_size = res_size * 4;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int)));
  CubDebugExit(cudaMemset(d_res, 0, res_array_size * sizeof(int)));
  int* h_res = new int[res_array_size];

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

  cudaEvent_t start1, stop1, stop2;
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);
  cudaEventCreate(&stop2);
  cout << "TIMES: " << TIMES << "[";
  for (int i = 0; i < TIMES; i++) {
    cudaEventRecord(start, 0);
    {
      int num_blocks = (LO_LEN - 1) / THREADS_PER_BLOCK + 1;
      create_BS_sample<<<num_blocks, THREADS_PER_BLOCK>>>(bs_lo_orderdate, bs_lo_custkey, bs_lo_suppkey, bs_lo_revenue, d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_revenue, curand_state);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&unit_time, start, stop);
    time1 += unit_time;
    if (i % 10 == 0) cout << i;
    cout << "=" << flush;
    cudaEventRecord(start, 0);
    {
      CubDebugExit(cudaMemset(d_res, 0, res_array_size * sizeof(int)));
      queryKernel(
          bs_lo_orderdate, bs_lo_custkey, bs_lo_suppkey, bs_lo_revenue,
          d_c_custkey, d_c_city,
          d_s_suppkey, d_s_city,
          d_d_datekey, d_d_year,
          d_ht_c, d_ht_s, d_ht_d, d_res);
      CubDebugExit(cudaMemcpy(h_res, d_res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < res_size; i++) {
        if (h_res[4*i] != 0) {
          // cout << "run: " << h_res[6*i] << " " << h_res[6*i + 1] << " " << h_res[6*i + 2] << " " << reinterpret_cast<unsigned long long*>(&h_res[6*i + 4])[0]  << endl;
          int c_nation = h_res[4*i];
          int s_nation = h_res[4*i + 1];
          int d_year = h_res[4*i + 2];
          int sum = h_res[4*i + 3];
          tuple<int, int, int> t(c_nation, s_nation, d_year);
          bs_umap[t].push_back(sum);
        }
      }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&unit_time, start, stop);
    time2 += unit_time;
  }
  cout << "]" << endl;
  cout << "Time Taken(resample): " << time1 << "ms" << endl;

  cudaEventRecord(start, 0);
  for (auto& entry : bs_umap) {
    vector<int>& vec = entry.second;
    sort(vec.begin(), vec.end());
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&unit_time, start, stop);
  time2 += unit_time;

  // 计算原始样本的query值
  CubDebugExit(cudaMemset(d_res, 0, res_array_size * sizeof(int)));
  queryKernel(
      d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_revenue,
      d_c_custkey, d_c_city,
      d_s_suppkey, d_s_city,
      d_d_datekey, d_d_year,
      d_ht_c, d_ht_s, d_ht_d, d_res);
  CubDebugExit(cudaMemcpy(h_res, d_res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost));
  int res_count = 0;
  for (int i = 0; i < res_size; i++) {
    if (h_res[4*i] != 0) {
      int c_nation = h_res[4*i];
      int s_nation = h_res[4*i + 1];
      int d_year = h_res[4*i + 2];
      int sum = h_res[4*i + 3];
      tuple<int, int, int> t(c_nation, s_nation, d_year);
      umap[t] = sum;
      res_count++;
    }
  }

  for (auto& entry1 : umap) {
    auto t1 = entry1.first;
    long long sum = entry1.second;
    if(bs_umap.find(t1) == bs_umap.end()){
      cout << get<0>(t1) << "\t" << get<1>(t1) << "\t" << get<2>(t1) << "\t" << sum << "(0,0)" << endl;
      continue;
    }
    vector<int> bs_sum = bs_umap[t1];
    int length = bs_sum.size();
    int idx1 = length * 0.01;
    int idx2 = length * 0.99;
    cout << get<0>(t1) << "\t" << get<1>(t1) << "\t" << get<2>(t1) << "\t";
    cout << sum << "(" << (double)(bs_sum[idx1]-sum)/sum << "," << (double)(bs_sum[idx2]-sum)/sum << ")" << endl;
  }
  cout << "Res Count: " << res_count << endl;
  cout << "Time Taken(run query): " << time2 << "ms" << endl;

  CLEANUP(curand_state);
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
  int *h_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);
  int *h_c_custkey = loadColumn<int>("c_custkey", C_LEN);
  int *h_c_city = loadColumn<int>("c_city", C_LEN);
  int *h_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
  int *h_s_city = loadColumn<int>("s_city", S_LEN);
  int *h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int *h_d_year = loadColumn<int>("d_year", D_LEN);


  // 注册st、finish时间点，c++的计时工具
	chrono::high_resolution_clock::time_point st, finish;
  st = chrono::high_resolution_clock::now();
  run(
      h_lo_orderdate, h_lo_custkey, h_lo_suppkey, h_lo_revenue,
      h_c_custkey, h_c_city,
      h_s_suppkey, h_s_city,
      h_d_datekey, h_d_year,
      g_allocator);
  finish = chrono::high_resolution_clock::now();
  chrono::duration<double> diff = finish - st;
  cout << "total time: " << diff.count() * 1000 << "ms" << endl;
  return 0;
}