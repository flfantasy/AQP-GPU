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
using namespace std;

static hyperapi::TableDefinition bootstrapTable{
  "lineorder_bs",
  {
    hyperapi::TableDefinition::Column{"lo_orderdate", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
    hyperapi::TableDefinition::Column{"lo_discount", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
    hyperapi::TableDefinition::Column{"lo_quantity", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
    hyperapi::TableDefinition::Column{"lo_extendedprice", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable}
  }
};

static void resample(string pathToDatabase){
  // load column data to host memory
  int *h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
  int *h_lo_discount = loadColumn<int>("lo_discount", LO_LEN);
  int *h_lo_quantity = loadColumn<int>("lo_quantity", LO_LEN);
  int *h_lo_extendedprice = loadColumn<int>("lo_extendedprice", LO_LEN);
  int *bs_lo_orderdate = new int[LO_LEN];
  int *bs_lo_discount = new int[LO_LEN];
  int *bs_lo_quantity = new int[LO_LEN];
  int *bs_lo_extendedprice = new int[LO_LEN];
  memset(bs_lo_orderdate, 0, sizeof(bs_lo_orderdate));
  memset(bs_lo_discount, 0, sizeof(bs_lo_discount));
  memset(bs_lo_quantity, 0, sizeof(bs_lo_quantity));
  memset(bs_lo_extendedprice, 0, sizeof(bs_lo_extendedprice));
  chrono::duration<double> total = chrono::seconds(0);
  // 随机数种子
  srand((unsigned)time(NULL));
  {
    hyperapi::HyperProcess hyper(hyperapi::Telemetry::SendUsageDataToTableau);
    {
      hyperapi::Connection connection(hyper.getEndpoint(), pathToDatabase);
      const hyperapi::Catalog& catalog = connection.getCatalog();
      cout << "query: q13" << endl;
      cout << "TIMES: " << TIMES << "[";
      for (size_t i = 0; i < TIMES; i++)
      {
        bootstrapTable.setTableName("q13_bs_" + to_string(i));
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
            bs_lo_discount[j] = h_lo_discount[loidx];
            bs_lo_quantity[j] = h_lo_quantity[loidx];
            bs_lo_extendedprice[j] = h_lo_extendedprice[loidx];
          }
          finish = chrono::high_resolution_clock::now();
          diff = finish - start;
          total += diff;
          // cout << i << "/" << TIMES << " resample time:" << diff.count() * 1000 << "ms" << endl;
          if (i % 10 == 0) cout << i;
          cout << "=" << flush;
          for (size_t j = 0; j < LO_LEN; j++)
          {
            inserter.addRow(bs_lo_orderdate[j], bs_lo_discount[j], bs_lo_quantity[j], bs_lo_extendedprice[j]);
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
  long long sum = 0;
  long long bs_sum[TIMES];
  memset(bs_sum, 0, sizeof(bs_sum));

  hyperapi::HyperProcess hyper(hyperapi::Telemetry::SendUsageDataToTableau);
	{
		hyperapi::Connection connection(hyper.getEndpoint(), pathToDatabase);
    hyperapi::Result rowsInTable = connection.executeQuery(
      "select sum(lo_extendedprice * lo_discount) as revenue "
      "from lineorder "
      "where lo_orderdate >= 19940204 "
      "and lo_orderdate <= 19940210 "
      "and lo_discount >= 5 "
      "and lo_discount <= 7 "
      "and lo_quantity >= 26 "
      "and lo_quantity <= 35;");
    auto& row = *(begin(rowsInTable));  // 和迭代器遍历等价的写法
    sum = (*(begin(row))).get<double>();
    rowsInTable.close();
    // 时间测试
    chrono::high_resolution_clock::time_point start, finish;
    chrono::duration<double> diff;
    start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < TIMES; i++){
      hyperapi::Result rowsInTable = connection.executeQuery(
        "select sum(lo_extendedprice * lo_discount) as revenue "
        "from q13_bs_" + to_string(i) + " "
        "where lo_orderdate >= 19940204 "
        "and lo_orderdate <= 19940210 "
        "and lo_discount >= 5 "
        "and lo_discount <= 7 "
        "and lo_quantity >= 26 "
        "and lo_quantity <= 35;");
      for (const hyperapi::Row& row : rowsInTable) {
        for (const hyperapi::Value& value : row) {
          bs_sum[i] = value.get<double>();
        }
      }
      rowsInTable.close();
    }
    sort(bs_sum, bs_sum + TIMES);
    finish = chrono::high_resolution_clock::now();
    diff = finish - start;

    cout << sum << "(" << (double)(bs_sum[1]-sum)/sum << "," << (double)(bs_sum[TIMES-2]-sum)/sum << ")" << endl;
    cout << "Time Taken(run query): " << diff.count() * 1000 << "ms" << endl;
  }
}

int main(int argc, char** argv) {
  const string pathToDatabase = "test/ssb/data/s1.hyper.q13";
  // 拷贝原始hyper文件
  system("cp /home/zhaoh/crystal/test/ssb/data/s1.hyper /home/zhaoh/crystal/test/ssb/data/s1.hyper.q13");
  resample(pathToDatabase);
  run(pathToDatabase);
  // 删除残留hyper文件
  system("rm /home/zhaoh/crystal/test/ssb/data/s1.hyper.q13");
  return 0;
}