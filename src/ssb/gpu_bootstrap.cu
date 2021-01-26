// Ensure printing of CUDA run1time errors to console
#define CUB_STDERR
#define TIMES 100
#define NUM_SM 72
#define WARPS_PER_SM 32
#define THREADS_PER_WARP 32
#define CONCURRENT_THREADS (NUM_SM * WARPS_PER_SM * THREADS_PER_WARP) 

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

__global__ void oldCurandGenKernel1(curandState *curand_states,long clock_for_rand) {
  curand_init(clock_for_rand, 0, 0, curand_states);
  // *max_sm = 0;
  // *min_sm = 10;
  // *max_warpid = 0;
  // *min_warpid = 10;
}

__global__ void oldKernel1(int* lo_revenue, int lo_num_entries, unsigned long long* sum,
      curandState *curand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < lo_num_entries * TIMES) {
    int randLO = curand(curand_states) % lo_num_entries;
    atomicAdd(&sum[x % TIMES], (unsigned long long)lo_revenue[randLO]);
  }
  // int smid = __mysmid();
  // int warpid = __mywarpid();
  // atomicMax(max_sm, smid);
  // atomicMin(min_sm, smid);
  // atomicMax(max_warpid, warpid);
  // atomicMin(min_warpid, warpid);
}

__global__ void  curandGenKernel1(curandState *curand_states,long clock_for_rand) {
  for(int i = 0; i < 72*32*32; i++) {
    curand_init(clock_for_rand + i, 0, 0, curand_states + i);
    // printf("%d: %d\n", i, (curand_states + i)->d);
  }
}

__global__ void kernel1(int* lo_revenue, int lo_num_entries, unsigned long long* sum,
    curandState *curand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < lo_num_entries * TIMES) {
    int smid = __mysmid();
    int warpid = __mywarpid();
    int laneid = __mylaneid();
    int randLO = curand(curand_states + (smid * 1024 + warpid * 32 + laneid)) % lo_num_entries;
    atomicAdd(&sum[x % TIMES], (unsigned long long)lo_revenue[randLO]);
  }
}

// vector
__global__ void kernel2(int* lo_revenue, int lo_num_entries, unsigned long long* sum,
    int entries_per_thread, curandState *curand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int threads_per_bootstrap = (lo_num_entries - 1) / entries_per_thread + 1;
  if (x < threads_per_bootstrap * TIMES){
    int smid = __mysmid();
    int warpid = __mywarpid();
    int laneid = __mylaneid();
    int curandid = smid * 1024 + warpid * 32 + laneid;
    int sumTemp = 0;
    if (x / TIMES == threads_per_bootstrap - 1){
      int entries = (lo_num_entries - 1) % entries_per_thread + 1;
      for(int i = 0; i < entries; i++){
        sumTemp += lo_revenue[curand(curand_states + curandid) % lo_num_entries];
      }
    } else {
      for(int i = 0; i < entries_per_thread; i++){
        sumTemp += lo_revenue[curand(curand_states + curandid) % lo_num_entries];
      }
    }
    atomicAdd(&sum[x % TIMES], (unsigned long long)sumTemp); 
  }
}

// bootstrap的方法1：atomicadd很多，sum数组传回CPU再排序
void run1(int* h_lo_revenue, int lo_num_entries, cub::CachingDeviceAllocator&  g_allocator,
  int threads_per_block, int entries_per_thread,
  unsigned long long* h_low_bound, unsigned long long* h_upper_bound) {
  // // 测试指针运算
  // int a[4];
  // a[0] = 0;  a[1] = 1;  a[2] = 2;  a[3] = 3;
  // for (int i = 0; i < 4; i++) {
  //   printf("%p\n", a+i);
  // }

  float time_query;
  // cuda注册start、stop事件，用于计时
  cudaEvent_t start, stop1, stop2, stop3, stop4, stop5;
  cudaEventCreate(&start);
  cudaEventCreate(&stop1);
  cudaEventCreate(&stop2);
  cudaEventCreate(&stop3);
  cudaEventCreate(&stop4);
  cudaEventCreate(&stop5);
  cudaEventRecord(start, 0);

  // load column data to device memory
  int *d_lo_revenue = loadToGPU<int>(h_lo_revenue, lo_num_entries, g_allocator);
  cudaEventRecord(stop1, 0);
  cudaEventSynchronize(stop1);
  cudaEventElapsedTime(&time_query, start, stop1);
  cout << "H2D时间:" << time_query << "ms" << endl;

  // 100次bootstarp采样
  unsigned long long* d_sum1 = NULL;
  unsigned long long* d_sum2 = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum1, 100 * sizeof(unsigned long long)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum2, 100 * sizeof(unsigned long long)));
  cudaMemset(d_sum1, 0, 100 * sizeof(unsigned long long));
  cudaMemset(d_sum2, 0, 100 * sizeof(unsigned long long));

  // 随机数生成器初始化，为了防止冲突需要设置多个。titan的并行度是72个sm * 32个warp/sm * 32个thread/warp
  curandState *curand_state;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&curand_state, 72*1024*sizeof(curandState)));
  long clock_for_rand = clock();  //程序运行时钟数

  // //测试smid范围和warpid范围
  // int* max_sm = NULL;
  // int* min_sm = NULL;
  // int* max_warpid = NULL;
  // int* min_warpid = NULL;
  // CubDebugExit(g_allocator.DeviceAllocate((void**)&max_sm, sizeof(int)));
  // CubDebugExit(g_allocator.DeviceAllocate((void**)&min_sm, sizeof(int)));
  // CubDebugExit(g_allocator.DeviceAllocate((void**)&max_warpid, sizeof(int)));
  // CubDebugExit(g_allocator.DeviceAllocate((void**)&min_warpid, sizeof(int)));
  curandGenKernel1<<<1, 1>>>(curand_state, clock_for_rand);    //main road

  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  cudaEventElapsedTime(&time_query, stop1, stop2);
  cout << "随机数生成器初始化:" << time_query << "ms" << endl;

  // // 统计smid和warpid的数量
  // unsigned int** sm = NULL;
  // CubDebugExit(g_allocator.DeviceAllocate((void***)&sm, 100 * sizeof(unsigned int*)));
  // for (int i = 0; i < 100; i++) {
  //   CubDebugExit(g_allocator.DeviceAllocate((void**)&sm[i], 500 * sizeof(unsigned int)));
  // }

  int num_blocks = (lo_num_entries * TIMES - 1) / threads_per_block + 1;
  // kernel1<<<num_blocks, threads_per_block>>>(d_lo_revenue, lo_num_entries, d_sum1, curand_state);    //main road

  cudaEventRecord(stop3, 0);
  cudaEventSynchronize(stop3);
  cudaEventElapsedTime(&time_query, stop2, stop3);
  cout << "GPU采样时间1:" << time_query << "ms" << endl;

  num_blocks = (((lo_num_entries - 1) / entries_per_thread + 1) * TIMES - 1) / threads_per_block + 1;
  kernel2<<<num_blocks, threads_per_block>>>(d_lo_revenue, lo_num_entries, d_sum2, entries_per_thread, curand_state);    //main road

  cudaEventRecord(stop4, 0);
  cudaEventSynchronize(stop4);
  cudaEventElapsedTime(&time_query, stop3, stop4);
  cout << "GPU采样时间2:" << time_query << "ms" << endl;

  // //测试smid范围和warpid范围
  // int* h_max_sm;
  // int* h_min_sm;
  // int* h_max_warpid;
  // int* h_min_warpid;
  // h_max_sm = (int*)malloc(sizeof(int));
  // h_min_sm = (int*)malloc(sizeof(int));
  // h_max_warpid = (int*)malloc(sizeof(int));
  // h_min_warpid = (int*)malloc(sizeof(int));
  // CubDebugExit(cudaMemcpy(h_max_sm, max_sm, sizeof(int), cudaMemcpyDeviceToHost));
  // CubDebugExit(cudaMemcpy(h_min_sm, min_sm, sizeof(int), cudaMemcpyDeviceToHost));
  // CubDebugExit(cudaMemcpy(h_max_warpid, max_warpid, sizeof(int), cudaMemcpyDeviceToHost));
  // CubDebugExit(cudaMemcpy(h_min_warpid, min_warpid, sizeof(int), cudaMemcpyDeviceToHost));
  // printf("max_sm: %d\n", *h_max_sm);
  // printf("min_sm: %d\n", *h_min_sm);
  // printf("max_warpid: %d\n", *h_max_warpid);
  // printf("min_warpid: %d\n", *h_min_warpid);
  unsigned long long h_sum1[100];
  CubDebugExit(cudaMemcpy(&h_sum1, d_sum1, 100 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  sort(h_sum1, h_sum1 + 100);
  unsigned long long h_sum2[100];
  CubDebugExit(cudaMemcpy(&h_sum2, d_sum2, 100 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  sort(h_sum2, h_sum2 + 100);
  cout << h_sum1[1] << "," << h_sum1[98] << endl;
  cout << h_sum2[1] << "," << h_sum2[98] << endl;
  *h_low_bound = h_sum2[1];
  *h_upper_bound = h_sum2[98];

  cudaEventRecord(stop5, 0);
  cudaEventSynchronize(stop5);
  cudaEventElapsedTime(&time_query, stop4, stop5);
  // cout << "D2H并排序时间:" << time_query << "ms" << endl;
  CLEANUP(d_sum1);
  CLEANUP(d_sum2);
  CLEANUP(curand_state);
  CLEANUP(d_lo_revenue);
}

__global__ void kernel(unsigned long long* sum) {
  int count = 0;
  for(int i = 0; i < 5; i++){
    count++;
  }
  atomicAdd(sum, (unsigned long long)count);
}

/**
 * Main
 */
int main(int argc, char** argv)
{
  // cudaFree(0);
  // unsigned long long* d_sum1 = NULL;
  // CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum1, sizeof(unsigned long long)));
	// chrono::high_resolution_clock::time_point st1, finish1;
  // st1 = chrono::high_resolution_clock::now();
  // // kernel<<<72*8*32,4>>>(d_sum1);
  // kernel<<<72*8,128>>>(d_sum1);

  // finish1 = chrono::high_resolution_clock::now();
  // chrono::duration<double> diff1 = finish1 - st1;
  // unsigned long long h_sum1;
  // CubDebugExit(cudaMemcpy(&h_sum1, d_sum1, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  // cout << "总时间: " << diff1.count() * 1000 << "ms" << "sum: " << h_sum1 << endl;
  // st1 = chrono::high_resolution_clock::now();
  // // kernel<<<72*8,128>>>(d_sum1);
  // kernel<<<72*8*32,4>>>(d_sum1);
  // finish1 = chrono::high_resolution_clock::now();
  // diff1 = finish1 - st1;
  // CubDebugExit(cudaMemcpy(&h_sum1, d_sum1, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  // cout << "总时间: " << diff1.count() * 1000 << "ms" << "sum: " << h_sum1 << endl;
  // return 0;

  // Initialize command line
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  // load column data to host memory
  int *h_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);
  // 计算原始样本sum
  long long sum = 0;
  for (int i = 0; i < LO_LEN; i++) {
    sum += h_lo_revenue[i];
  }

  // 置信区间结果
  unsigned long long low_bound = 0;
  unsigned long long upper_bound = 0;
  
  //
  int x = 16;
  int y = 100;
  int threads_per_block[x];
  for(int i = 0; i < x; i++){
    threads_per_block[i] = (i + 1) * 2;
  }
  int samples_per_thread[y];
  for(int i = 0; i < y; i++){
    samples_per_thread[i] = i + 1;
  }
  int time[x][y];

  // 注册st、finish时间点，c++的计时工具
	chrono::high_resolution_clock::time_point st, finish;
  for(int i = 0; i < x; i++){
    for(int j = 0; j < y; j++){
      cout << threads_per_block[i] << " " << samples_per_thread[j] << endl;
      st = chrono::high_resolution_clock::now();
      run1(h_lo_revenue, LO_LEN, g_allocator, threads_per_block[i], samples_per_thread[j], &low_bound, &upper_bound);  //main road
      finish = chrono::high_resolution_clock::now();
      chrono::duration<double> diff = finish - st;
      cout << "总时间: " << diff.count() * 1000 << "ms" << endl;
      time[i][j] = diff.count() * 1000;
    }
  }
  for(int i = 0; i < x; i++){
    for(int j = 0; j < y; j++){
      cout << time[i][j] << " ";
    }
    cout << endl;
  }
  cout << sum << "(" << (long)low_bound - sum << "," << upper_bound - sum << ")" << endl;
  cout << sum << "(" << (double)((long)low_bound - sum)/sum << "," << (double)(upper_bound - sum)/sum << ")" << endl;

  // st = chrono::high_resolution_clock::now();
  // run1(h_lo_revenue, LO_LEN, g_allocator, &low_bound, &upper_bound);
  // finish = chrono::high_resolution_clock::now();
  // std::chrono::duration<double> diff = finish - st;
  // cout << "方法二时间: " << diff.count() * 1000 << endl;
  // cout << sum << "(" << low_bound << "," << upper_bound << ")" << endl;

	return 0;
} 