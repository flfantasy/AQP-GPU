#define TIMES 128

#include <string>
#include <time.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <chrono>
#include <unordered_set>
#include <hyperapi/hyperapi.hpp>


static void runReadAndPrintDataFromExistingHyperFile() {
  const std::string pathToDatabase = "../data/s1.hyper";
  std::chrono::high_resolution_clock::time_point start, finish;
  std::chrono::duration<double> diff;
  {
    hyperapi::HyperProcess hyper(hyperapi::Telemetry::SendUsageDataToTableau);
    {
      hyperapi::Connection connection(hyper.getEndpoint(), pathToDatabase);
      const hyperapi::Catalog& catalog = connection.getCatalog();

      std::unordered_set<hyperapi::TableName> tableNames = catalog.getTableNames("public");
      for (auto& tableName : tableNames) {
        hyperapi::TableDefinition tableDefinition = catalog.getTableDefinition(tableName);
        std::cout << "Table " << tableName << " has qualified name: " << tableDefinition.getTableName() << std::endl;
        for (auto& column : tableDefinition.getColumns()) {
          std::cout << "\t Column " << column.getName() << " has type " << column.getType() << " and nullability " << column.getNullability()
                    << std::endl;
        }
        std::cout << "These are all rows in the table " << tableName.toString() << ":" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        hyperapi::Result rowsInTable1 = connection.executeQuery("select count(*) from " + tableName.toString());
        finish = std::chrono::high_resolution_clock::now();
        diff = finish - start;
        std::cout << "Time Taken Total: " << diff.count() * 1000 << "ms" << std::endl;
        for (const hyperapi::Row& row : rowsInTable1) {
          for (const hyperapi::Value& value : row) {
            std::cout << value << '\t';
          }
          std::cout << '\n';
        }
      }
    }
  }
}


int main(int argc, char** argv) {
   try {
      runReadAndPrintDataFromExistingHyperFile();
   } catch (const hyperapi::HyperException& e) {
      std::cout << e.toString() << std::endl;
      return 1;
   }
  return 0;
}
