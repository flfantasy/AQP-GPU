#define TIMES 128

#include <iostream>
#include <string.h>
#include <algorithm>
#include <cstdlib>
#include <time.h>
#include <chrono>
#include "ssb_utils.h"

using namespace std;

void run(int* h_lo_orderdate, int* h_lo_discount, int* h_lo_quantity, int* h_lo_extendedprice, long long* sum) {
  // random seed
  srand((unsigned)time(NULL));
  // TIMES次bootstrap采样
  for (int i = 0; i < TIMES; i++) {
    // 每次采样LO_LEN条数据
    sum[i] = 0;
    for (int j = 0; j < LO_LEN; j++) {
      int loidx = rand() % LO_LEN;
      if (h_lo_orderdate[loidx] >= 19930101 && h_lo_orderdate[loidx] < 19940101 && h_lo_quantity[loidx] < 25 && h_lo_discount[loidx] >= 1 && h_lo_discount[loidx] <= 3){
        sum[i] += (unsigned long long)(h_lo_discount[loidx] * h_lo_extendedprice[loidx]);
      }
    }
  }
  sort(sum, sum + TIMES);
}

int main(int argc, char** argv) {
  // load column data to host memory
  int *h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
  int *h_lo_discount = loadColumn<int>("lo_discount", LO_LEN);
  int *h_lo_quantity = loadColumn<int>("lo_quantity", LO_LEN);
  int *h_lo_extendedprice = loadColumn<int>("lo_extendedprice", LO_LEN);

  // 原始样本的query值
  long long sum = 0;
  for (int i = 0; i < LO_LEN; i++) {
    if (h_lo_orderdate[i] >= 19930101 && h_lo_orderdate[i] < 19940101 && h_lo_quantity[i] < 25 && h_lo_discount[i] >= 1 && h_lo_discount[i] <= 3){
      sum += (unsigned long long)(h_lo_discount[i] * h_lo_extendedprice[i]);
    }
  }
  chrono::high_resolution_clock::time_point start, finish;
  chrono::duration<double> diff;

  // 时间测试
  start = chrono::high_resolution_clock::now();
  long long bs_sum[TIMES];
  memset(bs_sum, 0, sizeof(bs_sum));
  run(h_lo_orderdate, h_lo_discount, h_lo_quantity, h_lo_extendedprice, bs_sum);
  finish = chrono::high_resolution_clock::now();
  diff = finish - start;

  cout << sum << "(" << (double)(bs_sum[1]-sum)/sum << "," << (double)(bs_sum[TIMES-2]-sum)/sum << ")" << endl;
  cout << "Time Taken Total: " << diff.count() * 1000 << "ms" << endl;

  return 0;
}
