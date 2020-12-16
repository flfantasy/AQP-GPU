#pragma once

// 设置start、stop事件
#define SETUP_TIMING() cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

// 计时
#define TIME_FUNC(f,t) { \
    cudaEventRecord(start, 0); \
    f; \
    cudaEventRecord(stop, 0); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&t, start,stop); \
}

#define CLEANUP(vec) if(vec)CubDebugExit(g_allocator.DeviceFree(vec))

#define ALLOCATE(vec,size) CubDebugExit(g_allocator.DeviceAllocate((void**)&vec, size))

// 拷贝变量到显存
template<typename T>
T* loadToGPU(T* src, int numEntries, cub::CachingDeviceAllocator& g_allocator) {
  T* dest;
  // 分配设备内存空间
  CubDebugExit(g_allocator.DeviceAllocate((void**)&dest, sizeof(T) * numEntries));
  // copy to device
  CubDebugExit(cudaMemcpy(dest, src, sizeof(T) * numEntries, cudaMemcpyHostToDevice));
  return dest;
}

// 一个block的线程数 * 每个线程的items数
#define TILE_SIZE (BLOCK_THREADS * ITEMS_PER_THREAD)

// 打印报错信息
#define CHECK_ERROR() { \
  cudaDeviceSynchronize(); \
  cudaError_t error = cudaGetLastError(); \
  if(error != cudaSuccess) \
  { \
    printf("CUDA error: %s\n", cudaGetErrorString(error)); \
    exit(-1); \
  } \
}
