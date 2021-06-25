/*反向采样*/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR
#define TIMES 128
#define NUM 32
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

// // old version
// __global__ void kernel(int* lo_revenue, int lo_num_entries, unsigned long long* sum,
//     unsigned long long* count, curandState *curand_states) {
//   int x = blockIdx.x * blockDim.x + threadIdx.x;
//   if (x < lo_num_entries * TIMES) {
//     int smid = __mysmid();
//     int warpid = __mywarpid();
//     int laneid = __mylaneid();
//     unsigned int rand = curand(curand_states + (smid * 1024 + warpid * 32 + laneid));
//     int idx = rand % TIMES;
//     int loidx = x / TIMES;
//     atomicAdd(&count[idx], 1ULL);
//     atomicAdd(&sum[idx], (unsigned long long)lo_revenue[loidx]);
//     // atomicAdd(&sum[idx], (unsigned long long)loidx);
//     // count[idx] += 1;
//     // sum[idx] += (unsigned long long)lo_revenue[loidx];
//   }
// }

// new version
__global__ void kernel(int* lo_revenue, unsigned long long* sum,
    unsigned long long* count, curandState *curand_states,
    unsigned int* duration1, unsigned int* duration2, unsigned int* duration3) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  unsigned int start, stop1, stop2, stop3;
  start = clock();
  __shared__ unsigned long long s_sum[128];
  __shared__ unsigned long long s_count[128];
  s_sum[tid] = 0;
  s_count[tid] = 0;
  __shared__ int s_revenue[NUM];
  if(tid < NUM){
    int x = bid * NUM + tid;
    if(x < LO_LEN){
      s_revenue[tid] = lo_revenue[x];
    }
  }
  __syncthreads();
  stop1 = clock();

  int smid = __mysmid();
  int warpid = __mywarpid();
  int laneid = __mylaneid();
  curandState* curand_state = curand_states + (smid * 1024 + warpid * 32 + laneid);
  for (int i = 0; i < NUM; i++)
  {
    unsigned int rand = curand(curand_state);
    int idx = rand % 128;
    atomicAdd(&s_sum[idx], s_revenue[i]);
    atomicAdd(&s_count[idx], 1);
  }
  __syncthreads();
  stop2 = clock();

  atomicAdd(&sum[tid], s_sum[tid]);
  atomicAdd(&count[tid], s_count[tid]);
  __syncthreads();
  stop3 = clock();
  if(bid < 1000){
    duration1[bid * blockDim.x + tid] = stop1 - start;
    duration2[bid * blockDim.x + tid] = stop2 - stop1;
    duration3[bid * blockDim.x + tid] = stop3 - stop2;
  }
}

// bootstrap的方法1：每次采样都atomicadd，sum数组传回CPU再排序
void run(int* h_lo_revenue, int lo_num_entries, cub::CachingDeviceAllocator&  g_allocator,
  int threads_per_block, unsigned long long* h_low_bound, unsigned long long* h_upper_bound) {

  float time_query;
  // cuda注册start、stop事件，用于计时
  cudaEvent_t start, stop1, stop2, stop3, stop4;
{  cudaEventCreate(&start);
  cudaEventCreate(&stop1);
  cudaEventCreate(&stop2);
  cudaEventCreate(&stop3);
  cudaEventCreate(&stop4);
  cudaEventRecord(start, 0);}

  // 数据从CPU传输到GPU
  int *d_lo_revenue = loadToGPU<int>(h_lo_revenue, lo_num_entries, g_allocator);
{  cudaEventRecord(stop1, 0);
  cudaEventSynchronize(stop1);
  cudaEventElapsedTime(&time_query, start, stop1);
  cout << "H2D时间:" << time_query << "ms" << endl;}

  // TIMES次bootstarp的sum
  unsigned long long* d_sum = NULL;
  unsigned long long* d_count = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum, TIMES * sizeof(unsigned long long)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_count, TIMES * sizeof(unsigned long long)));
  cudaMemset(d_sum, 0, TIMES * sizeof(unsigned long long));
  cudaMemset(d_count, 0, TIMES * sizeof(unsigned long long));

  // 随机数生成器初始化，为了防止冲突需要设置多个。titan的并行度是72个sm * 32个warp/sm * 32个thread/warp
  curandState *curand_state;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&curand_state, MOST_CONCURRENT_THREADS * sizeof(curandState)));
  long clock_for_rand = clock();  //程序运行时钟数
  curandGenKernel<<<1, 1>>>(curand_state, clock_for_rand);    //main road
{  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  cudaEventElapsedTime(&time_query, stop1, stop2);
  cout << "随机数生成器初始化:" << time_query << "ms" << endl;}

  int num_blocks = (lo_num_entries - 1) / NUM + 1;
  // 计时
  unsigned int* d_duration1 = NULL;
  unsigned int* d_duration2 = NULL;
  unsigned int* d_duration3 = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_duration1, 128000 * sizeof(unsigned int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_duration2, 128000 * sizeof(unsigned int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_duration3, 128000 * sizeof(unsigned int)));
  cudaMemset(d_duration1, 0, 128000 * sizeof(unsigned int));
  cudaMemset(d_duration2, 0, 128000 * sizeof(unsigned int));
  cudaMemset(d_duration3, 0, 128000 * sizeof(unsigned int));
  kernel<<<num_blocks, 128>>>(d_lo_revenue, d_sum, d_count, curand_state, d_duration1, d_duration2, d_duration3);    //main road
{  cudaEventRecord(stop3, 0);
  cudaEventSynchronize(stop3);
  cudaEventElapsedTime(&time_query, stop2, stop3);
  cout << "GPU采样时间:" << time_query << "ms" << endl;}

  // 计时
  unsigned int h_duration1[128000];
  unsigned int h_duration2[128000];
  unsigned int h_duration3[128000];
  CubDebugExit(cudaMemcpy(&h_duration1, d_duration1, 128000 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CubDebugExit(cudaMemcpy(&h_duration2, d_duration2, 128000 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CubDebugExit(cudaMemcpy(&h_duration3, d_duration3, 128000 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  sort(h_duration1, h_duration1 + 128000);
  sort(h_duration2, h_duration2 + 128000);
  sort(h_duration3, h_duration3 + 128000);
  unsigned long long tavg1 = 0, tavg2 = 0, tavg3 = 0;
  for (int i = 0; i < 128000; i++)
  {
    tavg1 += h_duration1[i];
    tavg2 += h_duration2[i];
    tavg3 += h_duration3[i];
  }
  tavg1 /= 128000;
  tavg2 /= 128000;
  tavg3 /= 128000;
  cout << "duration1 is " << tavg1 << "(" << h_duration1[0] << "," << h_duration1[128000-1] << ")" << endl;
  cout << "duration2 is " << tavg2 << "(" << h_duration2[0] << "," << h_duration2[128000-1] << ")" << endl;
  cout << "duration3 is " << tavg3 << "(" << h_duration3[0] << "," << h_duration3[128000-1] << ")" << endl;

{  unsigned long long h_sum[TIMES];
  unsigned long long h_count[TIMES];
  CubDebugExit(cudaMemcpy(&h_sum, d_sum, TIMES * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  CubDebugExit(cudaMemcpy(&h_count, d_count, TIMES * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  for(int i = 0; i < TIMES; i++)
    h_sum[i] = h_sum[i] * ((double)LO_LEN / h_count[i]);
  sort(h_sum, h_sum + TIMES);
  cout << h_sum[1] << "," << h_sum[TIMES-2] << endl;
  *h_low_bound = h_sum[1];
  *h_upper_bound = h_sum[TIMES-2];
  cudaEventRecord(stop4, 0);
  cudaEventSynchronize(stop4);
  cudaEventElapsedTime(&time_query, stop3, stop4);
  cout << "D2H并排序时间:" << time_query << "ms" << endl;
  CLEANUP(d_sum);
  CLEANUP(curand_state);
  CLEANUP(d_lo_revenue);}
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
  int *h_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);
    printf("%d\n", h_lo_revenue[LO_LEN-2]);
  // 计算原始样本sum
  long long sum = 0;
  for (int i = 0; i < LO_LEN; i++) {
    sum += h_lo_revenue[i];
  }

  // 置信区间结果
  unsigned long long low_bound = 0;
  unsigned long long upper_bound = 0;

  // 注册st、finish时间点，c++的计时工具
	chrono::high_resolution_clock::time_point st, finish;
  int threads_per_block = 32;
  st = chrono::high_resolution_clock::now();
  run(h_lo_revenue, LO_LEN, g_allocator, threads_per_block, &low_bound, &upper_bound);  //main road
  finish = chrono::high_resolution_clock::now();
  chrono::duration<double> diff = finish - st;
  cout << "总时间: " << diff.count() * 1000 << "ms" << endl;
  cout << sum << "(" << (long)low_bound - sum << "," << (long)upper_bound - sum << ")" << endl;
  cout << sum << "(" << (double)((long)low_bound - sum)/sum << "," << (double)(upper_bound - sum)/sum << ")" << endl;

	return 0;
} 