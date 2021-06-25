#pragma once

// 将block_itr数组中对应本线程处理的item存入items数组中
template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirect(
    const unsigned int tid,
    T* block_itr,       // tile数组指针
    T  (&items)[ITEMS_PER_THREAD]     // 目标数组
    ) {
  // 该线程对应的tile数组起始地址
  T* thread_itr = block_itr + tid;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
  }
}

// 将block_itr数组中对应本线程处理的item存入items数组中
template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirect(
    const unsigned int tid,
    T* block_itr,
    T  (&items)[ITEMS_PER_THREAD],
    int num_items       // tile的items数量，当最后一个tile数量不足时调用
    ) {
  T* thread_itr = block_itr + tid;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
    }
  }
}

// 根据tile容量是否是一个标准tile的容量，调用不同的BlockLoadDirect
template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoad(
    T* inp,          // load的起始位置
    T  (&items)[ITEMS_PER_THREAD],    // 目标数组
    int num_items     //当前tile有多少item
    ) {
  T* block_itr = inp;

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockLoadDirect<T, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, block_itr, items);
  } else {
    BlockLoadDirect<T, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, block_itr, items, num_items);
  }
}

#if 0

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirect(
    int tid,
    T* block_itr,
    T  (&items)[ITEMS_PER_THREAD]
    ) {
  T* thread_itr = block_itr + tid;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
  }
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirect(
    int tid,
    T* block_itr,
    T  (&items)[ITEMS_PER_THREAD]
    int num_items
    ) {
  T* thread_itr = block_itr + tid;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
    }
  }
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoad(
    T* inp,
    T  (&items)[ITEMS_PER_THREAD]
    int num_items
    ) {
  T* block_itr = inp + blockIdx.x * blockDim.x;

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockLoadDirect(threadIdx.x, block_itr, items);
  } else {
    BlockLoadDirect(threadIdx.x, block_itr, items, num_items);
  }
}

#endif
