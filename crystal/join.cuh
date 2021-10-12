#pragma once

#define HASH(X,Y,Z) ((X-Z) % Y)

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_1(
    int tid,
    K  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      int hash = HASH(items[ITEM], ht_len, keys_min);

      K slot = ht[hash];
      if (slot != 0) {
        selection_flags[ITEM] = 1;
      } else {
        selection_flags[ITEM] = 0;
      }
    }
  }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_1(
    int tid,
    K  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int hash = HASH(items[ITEM], ht_len, keys_min);

        K slot = ht[hash];
        if (slot != 0) {
          selection_flags[ITEM] = 1;
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }
  }
}

// 有keys_min参数的版本
// 探测items中的元素是否保存在ht中，结果保存在selection_flags中
template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_1(
    K  (&items)[ITEMS_PER_THREAD],  // keys数组
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockProbeDirectAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, selection_flags, ht, ht_len, keys_min);
  } else {
    BlockProbeDirectAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, selection_flags, ht, ht_len, keys_min, num_items);
  }
}

// 有keys_min参数的版本，默认keys_min为0
// 探测items中的元素是否保存在ht中，结果保存在selection_flags中
template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_1(
    K  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    int num_items
    ) {
  BlockProbeAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, ht, ht_len, 0, num_items);
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_2(
    int tid,
    K  (&keys)[ITEMS_PER_THREAD],
    V  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      int hash = HASH(keys[ITEM], ht_len, keys_min);

      uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
      if (slot != 0) {
        res[ITEM] = (slot >> 32);
      } else {
        selection_flags[ITEM] = 0;
      }
    }
  }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_2(
    int tid,
    K  (&items)[ITEMS_PER_THREAD],
    V  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int hash = HASH(items[ITEM], ht_len, keys_min);

        uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
        if (slot != 0) {
          res[ITEM] = (slot >> 32);
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }
  }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_2(
    K  (&keys)[ITEMS_PER_THREAD],
    V  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockProbeDirectAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min);
  } else {
    BlockProbeDirectAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min, num_items);
  }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_2(
    K  (&keys)[ITEMS_PER_THREAD],
    V  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    int num_items
    ) {
  BlockProbeAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, res, selection_flags, ht, ht_len, 0, num_items);
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_1(
    int tid,
    K  (&keys)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      int hash = HASH(keys[ITEM], ht_len, keys_min);

      K old = atomicCAS(&ht[hash], 0, keys[ITEM]);
    }
  }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_1(
    int tid,
    K  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int hash = HASH(items[ITEM], ht_len, keys_min);

        K old = atomicCAS(&ht[hash], 0, items[ITEM]);
      }
    }
  }
}

// 有keys_min参数的版本
// 向ht中插入只有keys构成的KV对。
template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_1(
    K  (&keys)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {

  // keys数组是否是满的
  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockBuildDirectSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, selection_flags, ht, ht_len, keys_min);
  } else {
    BlockBuildDirectSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, selection_flags, ht, ht_len, keys_min, num_items);
  }
}

// 无keys_min参数的版本，默认keys_min为0
// 向ht中插入只有keys构成的KV对。
template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_1(
    K  (&keys)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    int num_items
    ) {
  BlockBuildSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, selection_flags, ht, ht_len, 0, num_items);
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_2(
    int tid,
    K  (&keys)[ITEMS_PER_THREAD],
    V  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      int hash = HASH(keys[ITEM], ht_len, keys_min);

      K old = atomicCAS(&ht[hash << 1], 0, keys[ITEM]);
      ht[(hash << 1) + 1] = res[ITEM];
    }
  }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_2(
    int tid,
    K  (&keys)[ITEMS_PER_THREAD],
    V  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int hash = HASH(keys[ITEM], ht_len, keys_min);

        K old = atomicCAS(&ht[hash << 1], 0, keys[ITEM]);
        ht[(hash << 1) + 1] = res[ITEM];
      }
    }
  }
}

// 有keys_min参数的版本
// 向ht中插入keys和res构成的KV对。
template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_2(
    K  (&keys)[ITEMS_PER_THREAD], // key的数组
    V  (&res)[ITEMS_PER_THREAD],  // value的数组
    int  (&selection_flags)[ITEMS_PER_THREAD], // 表示是否被select，选中则要插入ht
    K* ht,  // hash table
    int ht_len, // ht的slot数量
    K keys_min, // key的最小值
    int num_items // keys的实际数量
    ) {

  // keys数组是否是满的
  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockBuildDirectSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
        threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min);
  } else {
    BlockBuildDirectSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
        threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min, num_items);
  }
}

// 无keys_min参数的版本，默认keys_min为0
// 向ht中插入keys和res构成的KV对。
template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_2(
    K  (&keys)[ITEMS_PER_THREAD],
    V  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    int num_items
    ) {
  BlockBuildSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, res, selection_flags, ht, ht_len, 0, num_items);
}
