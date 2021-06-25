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
    hyperapi::TableDefinition::Column{"lo_custkey", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
    hyperapi::TableDefinition::Column{"lo_suppkey", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
    hyperapi::TableDefinition::Column{"lo_revenue", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable}
  }
};


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

static void resample(string pathToDatabase){
  // load column data to host memory
  int *h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
  int *h_lo_custkey = loadColumn<int>("lo_custkey", LO_LEN);
  int *h_lo_suppkey = loadColumn<int>("lo_suppkey", LO_LEN);
  int *h_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);
  int *bs_lo_orderdate = new int[LO_LEN];
  int *bs_lo_custkey = new int[LO_LEN];
  int *bs_lo_suppkey = new int[LO_LEN];
  int *bs_lo_revenue = new int[LO_LEN];
  memset(bs_lo_orderdate, 0, sizeof(bs_lo_orderdate));
  memset(bs_lo_custkey, 0, sizeof(bs_lo_custkey));
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
      cout << "query: q32" << endl;
      cout << "TIMES: " << TIMES << "[";
      for (size_t i = 0; i < TIMES; i++)
      {
        bootstrapTable.setTableName("q32_bs_" + to_string(i));
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
            bs_lo_custkey[j] = h_lo_custkey[loidx];
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
            inserter.addRow(bs_lo_orderdate[j], bs_lo_custkey[j], bs_lo_suppkey[j], bs_lo_revenue[j]);
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
  unordered_map<tuple<int, int, int>, long long, tuple_hash> umap;
  unordered_map<tuple<int, int, int>, vector<long long>, tuple_hash> bs_umap;

  hyperapi::HyperProcess hyper(hyperapi::Telemetry::SendUsageDataToTableau);
	{
		hyperapi::Connection connection(hyper.getEndpoint(), pathToDatabase);

    hyperapi::Result rowsInTable = connection.executeQuery(
      "select c_city, s_city, d_year, sum(lo_revenue) as revenue "
      "from lineorder, customer, supplier, ddate "
      "where lo_custkey = c_custkey "
      "and lo_suppkey = s_suppkey "
      "and lo_orderdate = d_datekey "
      "and c_nation = 24 "
      "and s_nation = 24 "
      "and d_year >= 1992 and d_year <= 1997 "
      "group by c_city, s_city, d_year;");
    for (const hyperapi::Row& row : rowsInTable) {
      hyperapi::ColumnIterator iterator = begin(row);
      int c_city = (*(iterator)).get<int>();
      int s_city = (*(++iterator)).get<int>();
      int d_year = (*(++iterator)).get<int>();
      long long sum = (*(++iterator)).get<double>();
      tuple<int, int, int> t(c_city, s_city, d_year);
      umap[t] = sum;
    }
    rowsInTable.close();

    // 时间测试
    chrono::high_resolution_clock::time_point start, finish;
    chrono::duration<double> diff;
    start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < TIMES; i++){
      hyperapi::Result rowsInTable = connection.executeQuery(
        "select c_city, s_city, d_year, sum(lo_revenue) as revenue "
        "from q32_bs_" + to_string(i) + ", customer, supplier, ddate "
        "where lo_custkey = c_custkey "
        "and lo_suppkey = s_suppkey "
        "and lo_orderdate = d_datekey "
        "and c_nation = 24 "
        "and s_nation = 24 "
        "and d_year >= 1992 and d_year <= 1997 "
        "group by c_city, s_city, d_year;");
      for (const hyperapi::Row& row : rowsInTable) {
        hyperapi::ColumnIterator iterator = begin(row);
        int c_city = (*(iterator)).get<int>();
        int s_city = (*(++iterator)).get<int>();
        int d_year = (*(++iterator)).get<int>();
        long long sum = (*(++iterator)).get<double>();
        tuple<int, int, int> t(c_city, s_city, d_year);
        bs_umap[t].push_back(sum);
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
      auto t1 = entry1.first;
      long long sum = entry1.second;
      if(bs_umap.find(t1) == bs_umap.end()){
        cout << get<0>(t1) << "\t" << get<1>(t1) << "\t" << get<2>(t1) << "\t" << sum << "(0,0)" << endl;
        continue;
      }
      vector<long long> bs_sum = bs_umap[t1];
      int length = bs_sum.size();
      int idx1 = length * 0.01;
      int idx2 = length * 0.99;
      cout << get<0>(t1) << "\t" << get<1>(t1) << "\t" << get<2>(t1) << "\t";
      cout << sum << "(" << (double)(bs_sum[idx1]-sum)/sum << "," << (double)(bs_sum[idx2]-sum)/sum << ")" << endl;
    }
    cout << "Time Taken(run query): " << diff.count() * 1000 << "ms" << endl;
  }
}

int main(int argc, char** argv) {
  const string pathToDatabase = "test/ssb/data/s1.hyper.q32";
  // 拷贝原始hyper文件
  system("cp /home/zhaoh/crystal/test/ssb/data/s1.hyper /home/zhaoh/crystal/test/ssb/data/s1.hyper.q32");
  resample(pathToDatabase);
  run(pathToDatabase);
  // 删除残留hyper文件
  system("rm /home/zhaoh/crystal/test/ssb/data/s1.hyper.q32");
  return 0;
}