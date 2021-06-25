#define TIMES 128

#include <string>
#include <time.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <unordered_set>
#include <hyperapi/hyperapi.hpp>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include "ssb_utils.h"
#include <unordered_map>
using namespace std;

static hyperapi::TableDefinition bootstrapTable{
  "lineorder_bs",
  {
    hyperapi::TableDefinition::Column{"lo_orderdate", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
    hyperapi::TableDefinition::Column{"lo_partkey", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
    hyperapi::TableDefinition::Column{"lo_suppkey", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
    hyperapi::TableDefinition::Column{"lo_revenue", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable}
  }
};


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

static void resample(string pathToDatabase){
  // load column data to host memory
  int *h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
  int *h_lo_partkey = loadColumn<int>("lo_partkey", LO_LEN);
  int *h_lo_suppkey = loadColumn<int>("lo_suppkey", LO_LEN);
  int *h_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);
  int *bs_lo_orderdate = new int[LO_LEN];
  int *bs_lo_partkey = new int[LO_LEN];
  int *bs_lo_suppkey = new int[LO_LEN];
  int *bs_lo_revenue = new int[LO_LEN];
  memset(bs_lo_orderdate, 0, sizeof(bs_lo_orderdate));
  memset(bs_lo_partkey, 0, sizeof(bs_lo_partkey));
  memset(bs_lo_suppkey, 0, sizeof(bs_lo_suppkey));
  memset(bs_lo_revenue, 0, sizeof(bs_lo_revenue));
  chrono::duration<double> total = chrono::seconds(0);
  // 随机数种子
  srand((unsigned)time(NULL));
  {
    hyperapi::HyperProcess hyper(hyperapi::Telemetry::SendUsageDataToTableau);
    {
      hyperapi::Connection connection(hyper.getEndpoint(), pathToDatabase);
      const hyperapi::Catalog& catalog = connection.getCatalog();
      cout << "query: q23" << endl;
      cout << "TIMES: " << TIMES << "[";
      for (size_t i = 0; i < TIMES; i++)
      {
        bootstrapTable.setTableName("q23_bs_" + to_string(i));
        if(catalog.hasTable(bootstrapTable.getTableName())){
          continue;
        }
        catalog.createTableIfNotExists(bootstrapTable);
        {
          hyperapi::Inserter inserter(connection, bootstrapTable);
          // 时间测试
          chrono::high_resolution_clock::time_point start, finish;
          chrono::duration<double> diff;
          start = chrono::high_resolution_clock::now();
          for (size_t j = 0; j < LO_LEN; j++){
            int loidx = rand() % LO_LEN;
            bs_lo_orderdate[j] = h_lo_orderdate[loidx];
            bs_lo_partkey[j] = h_lo_partkey[loidx];
            bs_lo_suppkey[j] = h_lo_suppkey[loidx];
            bs_lo_revenue[j] = h_lo_revenue[loidx];
          }
          finish = chrono::high_resolution_clock::now();
          diff = finish - start;
          total += diff;
          // cout << i << "/" << TIMES << " resample time:" << diff.count() * 1000 << "ms" << endl;
          if (i % 10 == 0) cout << i;
          cout << "=" << flush;
          for (size_t j = 0; j < LO_LEN; j++)
          {
            inserter.addRow(bs_lo_orderdate[j], bs_lo_partkey[j], bs_lo_suppkey[j], bs_lo_revenue[j]);
          }
          inserter.execute();
          inserter.close();
        }
      }
      cout << "]" << endl;
    }
  }
  cout << "Time Taken(resample): " << total.count() * 1000 << "ms" << endl;
}

static void run(string pathToDatabase){
	// 原始样本和bs样本的query值
  unordered_map<pair<int, int>, long long, pair_hash> umap;
  unordered_map<pair<int, int>, vector<long long>, pair_hash> bs_umap;

  hyperapi::HyperProcess hyper(hyperapi::Telemetry::SendUsageDataToTableau);
	{
		hyperapi::Connection connection(hyper.getEndpoint(), pathToDatabase);

    hyperapi::Result rowsInTable = connection.executeQuery(
      "select sum(lo_revenue), d_year, p_brand1 "
      "from lineorder, ddate, part, supplier "
      "where lo_orderdate = d_datekey "
      "and lo_partkey = p_partkey "
      "and lo_suppkey = s_suppkey "
      "and p_brand1 = 260 "
      "and s_region = 3 "
      "group by d_year, p_brand1;");
    for (const hyperapi::Row& row : rowsInTable) {
      hyperapi::ColumnIterator iterator = begin(row);
      long long sum = (*iterator).get<double>();
      int d_year = (*(++iterator)).get<int>();
      int p_brand1 = (*(++iterator)).get<int>();
      pair<int, int> p(d_year, p_brand1);
      umap[p] = sum;
    }
    rowsInTable.close();

    // 时间测试
    chrono::high_resolution_clock::time_point start, finish;
    chrono::duration<double> diff;
    start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < TIMES; i++){
      hyperapi::Result rowsInTable = connection.executeQuery(
        "select sum(lo_revenue), d_year, p_brand1 "
        "from q23_bs_" + to_string(i) + ", ddate, part, supplier "
        "where lo_orderdate = d_datekey "
        "and lo_partkey = p_partkey "
        "and lo_suppkey = s_suppkey "
        "and p_brand1 = 260 "
        "and s_region = 3 "
        "group by d_year, p_brand1;");
      for (const hyperapi::Row& row : rowsInTable) {
        hyperapi::ColumnIterator iterator = begin(row);
        long long sum = (*iterator).get<double>();
        int d_year = (*(++iterator)).get<int>();
        int p_brand1 = (*(++iterator)).get<int>();
        pair<int, int> p(d_year, p_brand1);
        bs_umap[p].push_back(sum);
      }
      rowsInTable.close();
    }

    for (auto& entry : bs_umap) {
      vector<long long>& vec = entry.second;
      sort(vec.begin(), vec.end());
    }
    finish = chrono::high_resolution_clock::now();
    diff = finish - start;

    for (auto& entry1 : umap) {
      auto p1 = entry1.first;
      long long sum = entry1.second;
      if(bs_umap.find(p1) == bs_umap.end()){
        cout << sum << "(0,0)\t" << p1.first << "\t" << p1.second << endl;
        continue;
      }
      vector<long long> bs_sum = bs_umap[p1];
      int length = bs_sum.size();
      int idx1 = length * 0.01;
      int idx2 = length * 0.99;
      cout << sum << "(" << (double)(bs_sum[idx1]-sum)/sum << "," << (double)(bs_sum[idx2]-sum)/sum << ")\t";
      cout << p1.first << "\t" << p1.second << endl;
    }
    cout << "Time Taken(run query): " << diff.count() * 1000 << "ms" << endl;
  }
}

int main(int argc, char** argv) {
  const string pathToDatabase = "test/ssb/data/s1.hyper.q23";
  // 拷贝原始hyper文件
  system("cp /home/zhaoh/crystal/test/ssb/data/s1.hyper /home/zhaoh/crystal/test/ssb/data/s1.hyper.q23");
  resample(pathToDatabase);
  run(pathToDatabase);
  // 删除残留hyper文件
  system("rm /home/zhaoh/crystal/test/ssb/data/s1.hyper.q23");
  return 0;
}