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
  if (x < LO_LEN) {
    int smid = __mysmid();
    int warpid = __mywarpid();
    int laneid = __mylaneid();
    unsigned int rand = curand(curand_states + (smid * 1024 + warpid * 32 + laneid));
    int loidx = rand % LO_LEN;
    bs_lo_orderdate[x] = d_lo_orderdate[loidx];
    bs_lo_discount[x] = d_lo_discount[loidx];
    bs_lo_quantity[x] = d_lo_quantity[loidx];
    bs_lo_extendedprice[x] = d_lo_extendedprice[loidx];    
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
  CubDebugExit(g_allocator.DeviceAllocate((void**)&bs_lo_orderdate, LO_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&bs_lo_discount, LO_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&bs_lo_quantity, LO_LEN * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&bs_lo_extendedprice, LO_LEN * sizeof(int)));
  cudaMemset(bs_lo_orderdate, 0, LO_LEN * sizeof(int));
  cudaMemset(bs_lo_discount, 0, LO_LEN * sizeof(int));
  cudaMemset(bs_lo_quantity, 0, LO_LEN * sizeof(int));
  cudaMemset(bs_lo_extendedprice, 0, LO_LEN * sizeof(int));
  int* h_bs_lo_orderdate = new int[LO_LEN];
  int* h_bs_lo_discount = new int[LO_LEN];
  int* h_bs_lo_quantity = new int[LO_LEN];
  int* h_bs_lo_extendedprice = new int[LO_LEN];

  // cuda注册start、stop事件，用于计时
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float time1 = 0.0f;
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

  cout << "TIMES: " << TIMES << "[";
  for (int i = 0; i < TIMES; i++) {
    cudaEventRecord(start, 0);
    {
      int num_blocks = (LO_LEN - 1) / THREADS_PER_BLOCK + 1;
      create_BS_sample<<<num_blocks, THREADS_PER_BLOCK>>>(bs_lo_orderdate, bs_lo_discount, bs_lo_quantity, bs_lo_extendedprice, d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, curand_state);
      CubDebugExit(cudaMemcpy(h_bs_lo_orderdate, bs_lo_orderdate, LO_LEN * sizeof(int), cudaMemcpyDeviceToHost));
      CubDebugExit(cudaMemcpy(h_bs_lo_discount, bs_lo_discount, LO_LEN * sizeof(int), cudaMemcpyDeviceToHost));
      CubDebugExit(cudaMemcpy(h_bs_lo_quantity, bs_lo_quantity, LO_LEN * sizeof(int), cudaMemcpyDeviceToHost));
      CubDebugExit(cudaMemcpy(h_bs_lo_extendedprice, bs_lo_extendedprice, LO_LEN * sizeof(int), cudaMemcpyDeviceToHost));
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&unit_time, start, stop);
    time1 += unit_time;
    if (i % 10 == 0) cout << i;
    cout << "=" << flush;
  }
  cout << "]" << endl;
  cout << "Time Taken(resample): " << time1 << "ms" << endl;

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