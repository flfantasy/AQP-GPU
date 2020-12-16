drop table lineorder;
drop table part;
drop table customer;
drop table supplier;
drop table ddate;

create table lineorder (
    lo_orderkey INTEGER,
    lo_linenumber INTEGER,
    lo_custkey INTEGER,
    lo_partkey INTEGER,
    lo_suppkey INTEGER,
    lo_orderdate INTEGER,
    lo_orderpriority VARCHAR(20),
    lo_shippriority INTEGER,
    lo_quantity INTEGER,
    lo_extendedprice INTEGER,
    lo_ordtotalprice INTEGER,
    lo_discount INTEGER,
    lo_revenue INTEGER,
    lo_supplycost INTEGER,
    lo_tax INTEGER,
    lo_commitdate INTEGER,
    lo_shipmode VARCHAR(10)
);
copy into lineorder from '/home/zhaoh/crystal/test/ssb/data/s20/lineorder.tbl' delimiters '|','|\n';

create table part (
    p_partkey INTEGER,
    p_name VARCHAR(30),
    p_mfgr INTEGER,
    p_category INTEGER,
    p_brand1 INTEGER,
    p_color VARCHAR(10),
    p_type VARCHAR(30),
    p_size INTEGER,
    p_container VARCHAR(20)
);
copy into part from '/home/zhaoh/crystal/test/ssb/data/s20/part.tbl.p' delimiters '|','|\n'; 

create table customer (
    c_custkey INTEGER,
    c_name VARCHAR(20),
    c_address VARCHAR(30),
    c_city INTEGER,
    c_nation INTEGER,
    c_region INTEGER,
    c_phone VARCHAR(20),
    c_mktsegment VARCHAR(20)
);
copy into customer from '/home/zhaoh/crystal/test/ssb/data/s20/customer.tbl.p' delimiters '|','|\n';

create table supplier (
    s_suppkey INTEGER,
    s_name VARCHAR(20),
    s_address VARCHAR(30),
    s_city INTEGER,
    s_nation INTEGER,
    s_region INTEGER,
    s_phone VARCHAR(20)
);
copy into supplier from '/home/zhaoh/crystal/test/ssb/data/s20/supplier.tbl.p' delimiters '|','|\n'; 

create table ddate (
    d_datekey INTEGER,
    d_date VARCHAR(20),
    d_dayofweek VARCHAR(10),
    d_month VARCHAR(10),
    d_year INTEGER,
    d_yearmonthnum INTEGER,
    d_yearmonth VARCHAR(10),
    d_daynuminweek INTEGER,
    d_daynuminmonth INTEGER,
    d_daynuminyear INTEGER,
    d_monthnuminyear INTEGER,
    d_weeknuminyear INTEGER,
    d_sellingseason VARCHAR(10),
    d_lastdayinweekfl INTEGER,
    d_lastdayinmonthfl INTEGER,
    d_holidayfl INTEGER,
    d_weekdayfl INTEGER
);
copy into ddate from '/home/zhaoh/crystal/test/ssb/data/s20/date.tbl' delimiters '|','|\n'; 
