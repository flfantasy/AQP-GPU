/* This file is generated by code_gen.py */
#ifndef __SCHEMA_H__
#define __SCHEMA_H__
	struct supplier {
		int s_suppkey;
		char s_name[25];
		char s_address[25];
		int s_cit;
		int s_nation;
		int s_region;
		char s_phone[15];
	};

	struct customer {
		int c_custkey;
		char c_name[25];
		char c_address[25];
		int c_city;
		int c_nation;
		int c_region;
		char c_phone[15];
		char c_mktsegment[10];
	};

	struct part {
		int p_partkey;
		char p_name[22];
		int p_mfgr;
		int p_category;
		int p_brand1;
		char p_color[11];
		char p_type[25];
		int p_size;
		char p_container[10];
	};

	struct ddate {
		int d_datekey;
		char d_date[18];
		char d_dayofweek[8];
		char d_month[9];
		int d_year;
		int d_yearmonthnum;
		char d_yearmonth[7];
		int d_daynuminweek;
		int d_daynuminmonth;
		int d_daynuminyear;
		int d_monthnuminyear;
		int d_weeknuminyear;
		char d_sellingseason[12];
		int d_lastdayinweekfl;
		int d_lastdayinmonthfl;
		int d_holidayfl;
		int d_weekdayfl;
	};

	struct lineorder {
		int lo_orderkey;
		int lo_linenumber;
		int lo_custkey;
		int lo_partkey;
		int lo_suppkey;
		int lo_orderdate;
		char lo_orderpriority[16];
		int lo_shippriority;
		int lo_quantity;
		int lo_extendedprice;
		int lo_ordtotalprice;
		int lo_discount;
		int lo_revenue;
		int lo_supplycost;
		int lo_tax;
		int lo_commitdate;
		char lo_shipmode[10];
	};

#endif
