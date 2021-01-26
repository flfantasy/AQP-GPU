#include <iostream>
#include <string.h>
#include <algorithm>
#include <cstdlib>
#include <time.h>
#include <chrono>
#include "ssb_utils.h"

using namespace std;

void method1(int* lo_revenue, long long* sum) {
  // bootstrap样本数组
  int **bootstrap_samples = new int *[100];
  for(int i = 0; i < 100; i++)
  {
      bootstrap_samples[i] = new int[LO_LEN];  
  }
  // random seed
  srand((unsigned)time(NULL));
  // 100次bootstrap采样
  for (int i = 0; i < 100; i++) {
    // 每次采样LO_LEN条数据
    for (int j = 0; j < LO_LEN; j++) {
      bootstrap_samples[i][j] = lo_revenue[rand() % LO_LEN];
    }
    sum[i] = 0;
    for (int j = 0; j < LO_LEN; j++) {
      sum[i] += bootstrap_samples[i][j];
    }
  }
  sort(sum, sum + 100);
}

void method2(int* lo_revenue, long long* sum) {
  // random seed
  srand((unsigned)time(NULL));
  // 100次bootstrap采样
  for (int i = 0; i < 100; i++) {
    // 每次采样LO_LEN条数据
    sum[i] = 0;
    for (int j = 0; j < LO_LEN; j++) {
      sum[i] += lo_revenue[rand() % LO_LEN];
    }
  }
  sort(sum, sum + 100);
}

int main(int argc, char** argv) {
  // load column data to host memory
  int *h_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);
  // 每个bootstrap样本的sum，最后一个是原始样本的sum
  long long sum[101];
  memset(sum, 0, sizeof(sum));
  for (int i = 0; i < LO_LEN; i++) {
    sum[100] += h_lo_revenue[i];
  }
  chrono::high_resolution_clock::time_point start, finish;
  chrono::duration<double> diff;

  // method1时间测试
  start = chrono::high_resolution_clock::now();
  method1(h_lo_revenue, sum);
  finish = chrono::high_resolution_clock::now();
  diff = finish - start;

  cout << sum[100] << "(" << (double)(sum[1]-sum[100])/sum[100] << "," << (double)(sum[98]-sum[100])/sum[100] << ")" << endl;
  cout << "Time Taken Total: " << diff.count() * 1000  << "ms" << endl;

  // method2时间测试
  start = chrono::high_resolution_clock::now();
  method2(h_lo_revenue, sum);
  finish = chrono::high_resolution_clock::now();
  diff = finish - start;

  cout << sum[100] << "(" << (double)(sum[1]-sum[100])/sum[100] << "," << (double)(sum[98]-sum[100])/sum[100] << ")" << endl;
  cout << "Time Taken Total: " << diff.count() * 1000 << "ms" << endl;

  return 0;
}
