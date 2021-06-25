#define TIMES 128

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cstdlib>

#include <hyperapi/hyperapi.hpp>
// 表结构
static const hyperapi::TableDefinition lineorderTable{
   "lineorder",
    {
      hyperapi::TableDefinition::Column{"lo_orderkey", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_linenumber", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_custkey", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_partkey", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_suppkey", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_orderdate", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_orderpriority", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_shippriority", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_quantity", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_extendedprice", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_ordtotalprice", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_discount", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_revenue", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_supplycost", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_tax", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_commitdate", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"lo_shopmode", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable}
    }
};

static const hyperapi::TableDefinition partTable{
   "part", 
    {
      hyperapi::TableDefinition::Column{"p_partkey", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"p_name", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"p_mfgr", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"p_category", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"p_brand1", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"p_color", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"p_type", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"p_size", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"p_container", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable}
    }
};
static const hyperapi::TableDefinition supplierTable{
   "supplier", 
    {
      hyperapi::TableDefinition::Column{"s_suppkey", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"s_name", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"s_address", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"s_city", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"s_nation", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"s_region", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"s_phone", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable}
    }
};
static const hyperapi::TableDefinition customerTable{
   "customer", 
    {
      hyperapi::TableDefinition::Column{"c_custkey", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"c_name", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"c_address", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"c_city", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"c_nation", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"c_region", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"c_phone", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"c_mktsegment", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable}
    }
};
static const hyperapi::TableDefinition ddateTable{
   "ddate", 
    {
      hyperapi::TableDefinition::Column{"d_datekey", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_date", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_dayofweek", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_month", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_year", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_yearmonthnum", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_yearmonth", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_daynuminweek", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_daynuminmonth", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_daynuminyear", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_monthnuminyear", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_weeknuminyear", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_sellingseasin", hyperapi::SqlType::text(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_lastdayinweekfl", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_lastdayinmonthfl", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_holidayfl", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable},
      hyperapi::TableDefinition::Column{"d_weekdayfl", hyperapi::SqlType::integer(), hyperapi::Nullability::NotNullable}
    }
};

static void runCreateHyperFileFromCSV() {
  // hyper不能指定row之间的分隔符，默认是换行符，所以需要先改文件去掉行尾的'|'
  system("bash transform.sh");

  const std::string pathToDatabase = "../data/s1.hyper";
  {
    hyperapi::HyperProcess hyper(hyperapi::Telemetry::SendUsageDataToTableau);
    {
      hyperapi::Connection connection(hyper.getEndpoint(), pathToDatabase, hyperapi::CreateMode::CreateAndReplace);
      const hyperapi::Catalog& catalog = connection.getCatalog();
      catalog.createTableIfNotExists(lineorderTable);
      catalog.createTableIfNotExists(partTable);
      catalog.createTableIfNotExists(supplierTable);
      catalog.createTableIfNotExists(customerTable);
      catalog.createTableIfNotExists(ddateTable);

      std::string pathToCSV;
      pathToCSV = "../data/s1/lineorder.tbl";
      // The parameters of the COPY command are documented in the Tableau Hyper SQL documentation
      // (https:#help.tableau.com/current/api/hyper_api/en-us/reference/sql/sql-copy.html).
      int64_t rowCount = connection.executeCommand(
        "COPY " + lineorderTable.getTableName().toString() + " from " + hyperapi::escapeStringLiteral(pathToCSV) +
        " with ( delimiter '|')");  // hyper不能指定row之间的分隔符，需要先改文件
      std::cout << "The number of rows in table " << lineorderTable.getTableName() << " is " << rowCount << "." << std::endl;
      pathToCSV = "../data/s1/part.tbl.p";
      rowCount = connection.executeCommand(
        "COPY " + partTable.getTableName().toString() + " from " + hyperapi::escapeStringLiteral(pathToCSV) +
        " with ( delimiter '|')");  // hyper不能指定row之间的分隔符，需要先改文件
      std::cout << "The number of rows in table " << partTable.getTableName() << " is " << rowCount << "." << std::endl;
      pathToCSV = "../data/s1/supplier.tbl.p";
      rowCount = connection.executeCommand(
        "COPY " + supplierTable.getTableName().toString() + " from " + hyperapi::escapeStringLiteral(pathToCSV) +
        " with ( delimiter '|')");  // hyper不能指定row之间的分隔符，需要先改文件
      std::cout << "The number of rows in table " << supplierTable.getTableName() << " is " << rowCount << "." << std::endl;
      pathToCSV = "../data/s1/customer.tbl.p";
      rowCount = connection.executeCommand(
        "COPY " + customerTable.getTableName().toString() + " from " + hyperapi::escapeStringLiteral(pathToCSV) +
        " with ( delimiter '|')");  // hyper不能指定row之间的分隔符，需要先改文件
      std::cout << "The number of rows in table " << customerTable.getTableName() << " is " << rowCount << "." << std::endl;
      pathToCSV = "../data/s1/date.tbl";
      rowCount = connection.executeCommand(
        "COPY " + ddateTable.getTableName().toString() + " from " + hyperapi::escapeStringLiteral(pathToCSV) +
        " with ( delimiter '|')");  // hyper不能指定row之间的分隔符，需要先改文件
      std::cout << "The number of rows in table " << ddateTable.getTableName() << " is " << rowCount << "." << std::endl;
    }
  }
  system("bash transform_back.sh");
}


int main(int argc, char** argv) {
   try {
      runCreateHyperFileFromCSV();
   } catch (const hyperapi::HyperException& e) {
      std::cout << e.toString() << std::endl;
      return 1;
   }
  return 0;
}
