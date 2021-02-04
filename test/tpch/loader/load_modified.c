#include <stdio.h>
#include <stdlib.h>
#include <error.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>
#include <linux/limits.h>
#include "include/schema.h"
#include "include/common.h"

static char delimiter = '|';

// fp路径为"tpch/data/s*/lineitem.tbl.p", outName = "LINEITEM"
void lineitem (FILE *fp, char *outName){
  struct lineitem tmp;
  struct columnHeader header;
  FILE * out[16];
  char buf[1024] = {0};
  char data [1024] = {0};
  // tupleNum是源文件中的行数，tupleUnit仅为中间变量，tupleCount是当前block中已处理的tuple数量，tupleRemain是源文件剩余未处理的行数
  long tupleNum = 0, tupleUnit = 0, tupleCount = 0, tupleRemain = 0;
  int i = 0, prev = 0,count=0;

  for(i = 0; i < 16; i++){
    char path[PATH_MAX] = {0};
    sprintf(path,"%s%d",outName,i);
    // 每列一个out
    out[i] = fopen(path, "w");
    if(!out[i]){
      printf("Failed to open %s\n",path);
      exit(-1);
    }
  }

  while(fgets(buf,sizeof(buf),fp) !=NULL)
    tupleNum ++;
  header.totalTupleNum = tupleNum;
  tupleRemain = tupleNum;
  if(tupleNum > BLOCKNUM)
    tupleUnit = BLOCKNUM;
  else
    tupleUnit = tupleNum;
  // 当前block的tuple数量
  header.tupleNum = tupleUnit;
  header.format = UNCOMPRESSED;
  header.blockId = 0;
  // block的总数
  header.blockTotal = (tupleNum + BLOCKNUM -1) / BLOCKNUM ;
  fseek(fp,0,SEEK_SET);

  while(fgets(buf,sizeof(buf),fp)!= NULL){
    int writeHeader = 0;
    tupleCount ++;
    // 超过BLOCKNUM，该开始下一个block
    if(tupleCount > BLOCKNUM){
      tupleCount = 1;
      tupleRemain -= BLOCKNUM;
      if (tupleRemain > BLOCKNUM)
        tupleUnit = BLOCKNUM;
      else
        tupleUnit = tupleRemain;
      header.tupleNum = tupleUnit;
      header.blockId ++;
      writeHeader = 1;
    }
    for(i = 0, prev = 0, count = 0; buf[i] !='\n'; i++){
      if (buf[i] == delimiter){
        memset(data,0,sizeof(data));
        // 得到某一列的数据
        strncpy(data,buf+prev,i-prev);
        prev = i+1;
        switch(count){
           case 0:
            // 处理 l_orderkey 列
            // 修改此部分注意：1、tmp的属性名 2、sizeof的类型 3、out的序号 4、data转换成列的类型
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[0]);
            }
            tmp.l_orderkey = strtol(data,NULL,10);
            fwrite(&(tmp.l_orderkey),sizeof(int),1,out[0]);
            break;
           case 1:
            // 处理 l_partkey 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header, sizeof(struct columnHeader), 1, out[1]);
            }
            tmp.l_partkey = strtol(data, NULL, 10);
            fwrite(&(tmp.l_partkey), sizeof(int), 1, out[1]);
            break;
           case 2:
            // 处理 l_suppkey 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header, sizeof(struct columnHeader), 1, out[2]);
            }
            tmp.l_suppkey = strtol(data, NULL, 10);
            fwrite(&(tmp.l_suppkey), sizeof(int), 1, out[2]);
            break;
           case 3:
            // 处理 l_linenumber 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[3]);
            }
            tmp.l_linenumber = strtol(data, NULL, 10);
            fwrite(&(tmp.l_linenumber),sizeof(int), 1, out[3]);
            break;
           case 4:
            // 处理 l_quantity 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[4]);
            }
            tmp.l_quantity = strtol(data, NULL, 10);
            fwrite(&(tmp.l_quantity),sizeof(int), 1, out[4]);
            break;
           case 5:
            // 处理 l_extendedprice 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(float);
              fwrite(&header,sizeof(struct columnHeader),1,out[5]);
            }
            tmp.l_extendedprice = strtod(data, NULL);
            fwrite(&(tmp.l_extendedprice),sizeof(float), 1, out[5]);
            break;
           case 6:
            // 处理 l_discount 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(float);
              fwrite(&header,sizeof(struct columnHeader),1,out[6]);
            }
            tmp.l_discount = strtod(data, NULL);
            fwrite(&(tmp.l_discount),sizeof(float), 1, out[6]);
            break;
           case 7:
            // 处理 l_tax 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(float);
              fwrite(&header,sizeof(struct columnHeader),1,out[7]);
            }
            tmp.l_tax = strtod(data, NULL);
            fwrite(&(tmp.l_tax),sizeof(float), 1, out[7]);
            break;
           case 8:
            // 处理 l_returnflag 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[8]);
            }
            tmp.l_returnflag = strtol(data, NULL, 10);
            fwrite(&(tmp.l_returnflag),sizeof(int), 1, out[8]);
            break;
           case 9:
            // 处理 l_linestatus 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[9]);
            }
            tmp.l_linestatus = strtol(data, NULL, 10);
            fwrite(&(tmp.l_linestatus),sizeof(int), 1, out[9]);
            break;
           case 10:
            // 处理 l_shipdate 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[10]);
            }
            tmp.l_shipdate = strtol(data, NULL, 10);
            fwrite(&(tmp.l_shipdate),sizeof(int), 1, out[10]);
            break;
           case 11:
            // 处理 l_commitdate 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.l_commitdate);
              fwrite(&header,sizeof(struct columnHeader),1,out[11]);
            }
            strcpy(tmp.l_commitdate,data);
            fwrite(&(tmp.l_commitdate), sizeof(tmp.l_commitdate), 1, out[11]);
            break;
           case 12:
            // 处理 l_receiptdate 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.l_receiptdate);
              fwrite(&header,sizeof(struct columnHeader),1,out[12]);
            }
            strcpy(tmp.l_receiptdate,data);
            fwrite(&(tmp.l_receiptdate),sizeof(tmp.l_receiptdate), 1, out[12]);
            break;
           case 13:
            // 处理 l_shipinstruct 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.l_shipinstruct);
              fwrite(&header,sizeof(struct columnHeader),1,out[13]);
            }
            strcpy(tmp.l_shipinstruct,data);
            fwrite(&(tmp.l_shipinstruct),sizeof(tmp.l_shipinstruct), 1, out[13]);
            break;
           case 14:
            // 处理 l_shipmode 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.l_shipmode);
              fwrite(&header,sizeof(struct columnHeader),1,out[14]);
            }
            strcpy(tmp.l_shipmode,data);
            fwrite(&(tmp.l_shipmode),sizeof(tmp.l_shipmode), 1, out[14]);
            break;
           case 15:
            // 处理 l_comment 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.l_comment);
              fwrite(&header,sizeof(struct columnHeader),1,out[15]);
            }
            strcpy(tmp.l_comment,data);
            fwrite(&(tmp.l_comment),sizeof(tmp.l_comment), 1, out[15]);
            break;

        }
        count++;
      }
    }
    // 为了防止以"\n"作为行间隔造成最后一列未处理。正常应该以"|\n"作为行间隔。
    if(count == 15){
      if(writeHeader == 1){
        header.blockSize = header.tupleNum * sizeof(tmp.l_comment);
        fwrite(&header,sizeof(struct columnHeader),1,out[15]);
      }
      strncpy(tmp.l_comment,buf+prev,i-prev);
      fwrite(&(tmp.l_comment),sizeof(tmp.l_comment), 1, out[15]);
    }
  }

  for(i=0;i<16;i++){
    fclose(out[i]);
  }
}

// fp路径为"tpch/data/s*/orders.tbl.p", outName = "ORDERS"
void orders (FILE *fp, char *outName){
  struct orders tmp;
  struct columnHeader header;
  FILE * out[9];
  char buf[1024] = {0};
  char data [1024] = {0};
  // tupleNum是源文件中的行数，tupleUnit仅为中间变量，tupleCount是当前block中已处理的tuple数量，tupleRemain是源文件剩余未处理的行数
  long tupleNum = 0, tupleUnit = 0, tupleCount = 0, tupleRemain = 0;
  int i = 0, prev = 0,count=0;

  for(i = 0; i < 9; i++){
    char path[PATH_MAX] = {0};
    sprintf(path,"%s%d",outName,i);
    // 每列一个out
    out[i] = fopen(path, "w");
    if(!out[i]){
      printf("Failed to open %s\n",path);
      exit(-1);
    }
  }

  while(fgets(buf,sizeof(buf),fp) !=NULL)
    tupleNum ++;
  header.totalTupleNum = tupleNum;
  tupleRemain = tupleNum;
  if(tupleNum > BLOCKNUM)
    tupleUnit = BLOCKNUM;
  else
    tupleUnit = tupleNum;
  // 当前block的tuple数量
  header.tupleNum = tupleUnit;
  header.format = UNCOMPRESSED;
  header.blockId = 0;
  // block的总数
  header.blockTotal = (tupleNum + BLOCKNUM -1) / BLOCKNUM ;
  fseek(fp,0,SEEK_SET);

  while(fgets(buf,sizeof(buf),fp)!= NULL){
    int writeHeader = 0;
    tupleCount ++;
    // 超过BLOCKNUM，该开始下一个block
    if(tupleCount > BLOCKNUM){
      tupleCount = 1;
      tupleRemain -= BLOCKNUM;
      if (tupleRemain > BLOCKNUM)
        tupleUnit = BLOCKNUM;
      else
        tupleUnit = tupleRemain;
      header.tupleNum = tupleUnit;
      header.blockId ++;
      writeHeader = 1;
    }
    for(i = 0, prev = 0, count = 0; buf[i] !='\n'; i++){
      if (buf[i] == delimiter){
        memset(data,0,sizeof(data));
        // 得到某一列的数据
        strncpy(data,buf+prev,i-prev);
        prev = i+1;
        switch(count){
           case 0:
            // 处理 o_orderkey 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[0]);
            }
            tmp.o_orderkey = strtol(data,NULL,10);
            fwrite(&(tmp.o_orderkey),sizeof(int),1,out[0]);
            break;
           case 1:
            // 处理 o_custkey 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header, sizeof(struct columnHeader), 1, out[1]);
            }
            tmp.o_custkey = strtol(data, NULL, 10);
            fwrite(&(tmp.o_custkey), sizeof(int), 1, out[1]);
            break;
           case 2:
            // 处理 o_orderstatus 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.o_orderstatus);
              fwrite(&header, sizeof(struct columnHeader), 1, out[2]);
            }
            strcpy(tmp.o_orderstatus, data);
            fwrite(&(tmp.o_orderstatus), sizeof(tmp.o_orderstatus), 1, out[2]);
            break;
           case 3:
            // 处理 o_totalprice 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(float);
              fwrite(&header,sizeof(struct columnHeader),1,out[3]);
            }
            tmp.o_totalprice = strtod(data, NULL);
            fwrite(&(tmp.o_totalprice),sizeof(float), 1, out[3]);
            break;
           case 4:
            // 处理 o_orderdate 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[4]);
            }
            tmp.o_orderdate = strtol(data, NULL, 10);
            fwrite(&(tmp.o_orderdate),sizeof(int), 1, out[4]);
            break;
           case 5:
            // 处理 o_orderpriority 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.o_orderpriority);
              fwrite(&header,sizeof(struct columnHeader),1,out[5]);
            }
            strcpy(tmp.o_orderpriority, data);
            fwrite(&(tmp.o_orderpriority),sizeof(tmp.o_orderpriority), 1, out[5]);
            break;
           case 6:
            // 处理 o_clerk 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.o_clerk);
              fwrite(&header,sizeof(struct columnHeader),1,out[6]);
            }
            strcpy(tmp.o_clerk, data);
            fwrite(&(tmp.o_clerk),sizeof(tmp.o_clerk), 1, out[6]);
            break;
           case 7:
            // 处理 o_shippriority 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[7]);
            }
            tmp.o_shippriority = strtol(data, NULL, 10);
            fwrite(&(tmp.o_shippriority),sizeof(int), 1, out[7]);
            break;
           case 8:
            // 处理 o_comment 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.o_comment);
              fwrite(&header,sizeof(struct columnHeader),1,out[8]);
            }
            strcpy(tmp.o_comment, data);
            fwrite(&(tmp.o_comment),sizeof(tmp.o_comment), 1, out[8]);
            break;
        }
        count++;
      }
    }
    // 为了防止以"\n"作为行间隔造成最后一列未处理。正常应该以"|\n"作为行间隔。
    if(count == 8){
      if(writeHeader == 1){
        header.blockSize = header.tupleNum * sizeof(tmp.o_comment);
        fwrite(&header,sizeof(struct columnHeader),1,out[8]);
      }
      strncpy(tmp.o_comment,buf+prev,i-prev);
      fwrite(&(tmp.o_comment),sizeof(tmp.o_comment), 1, out[8]);
    }
  }

  for(i=0;i<9;i++){
    fclose(out[i]);
  }
}

// fp路径为"tpch/data/s*/customer.tbl.p", outName = "CUSTOMER"
void customer (FILE *fp, char *outName){
  struct customer tmp;
  struct columnHeader header;
  FILE * out[8];
  char buf[1024] = {0};
  char data [1024] = {0};
  // tupleNum是源文件中的行数，tupleUnit仅为中间变量，tupleCount是当前block中已处理的tuple数量，tupleRemain是源文件剩余未处理的行数
  long tupleNum = 0, tupleUnit = 0, tupleCount = 0, tupleRemain = 0;
  int i = 0, prev = 0,count=0;

  for(i = 0; i < 8; i++){
    char path[PATH_MAX] = {0};
    sprintf(path,"%s%d",outName,i);
    // 每列一个out
    out[i] = fopen(path, "w");
    if(!out[i]){
      printf("Failed to open %s\n",path);
      exit(-1);
    }
  }

  while(fgets(buf,sizeof(buf),fp) !=NULL)
    tupleNum ++;
  header.totalTupleNum = tupleNum;
  tupleRemain = tupleNum;
  if(tupleNum > BLOCKNUM)
    tupleUnit = BLOCKNUM;
  else
    tupleUnit = tupleNum;
  // 当前block的tuple数量
  header.tupleNum = tupleUnit;
  header.format = UNCOMPRESSED;
  header.blockId = 0;
  // block的总数
  header.blockTotal = (tupleNum + BLOCKNUM -1) / BLOCKNUM ;
  fseek(fp,0,SEEK_SET);

  while(fgets(buf,sizeof(buf),fp)!= NULL){
    int writeHeader = 0;
    tupleCount ++;
    // 超过BLOCKNUM，该开始下一个block
    if(tupleCount > BLOCKNUM){
      tupleCount = 1;
      tupleRemain -= BLOCKNUM;
      if (tupleRemain > BLOCKNUM)
        tupleUnit = BLOCKNUM;
      else
        tupleUnit = tupleRemain;
      header.tupleNum = tupleUnit;
      header.blockId ++;
      writeHeader = 1;
    }
    for(i = 0, prev = 0, count = 0; buf[i] !='\n'; i++){
      if (buf[i] == delimiter){
        memset(data,0,sizeof(data));
        // 得到某一列的数据
        strncpy(data,buf+prev,i-prev);
        prev = i+1;
        switch(count){
           case 0:
            // 处理 c_custkey 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[0]);
            }
            tmp.c_custkey = strtol(data,NULL,10);
            fwrite(&(tmp.c_custkey),sizeof(int),1,out[0]);
            break;
           case 1:
            // 处理 c_name 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.c_name);
              fwrite(&header, sizeof(struct columnHeader), 1, out[1]);
            }
            strcpy(tmp.c_name, data);
            fwrite(&(tmp.c_name), sizeof(tmp.c_name), 1, out[1]);
            break;
           case 2:
            // 处理 c_address 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.c_address);
              fwrite(&header, sizeof(struct columnHeader), 1, out[2]);
            }
            strcpy(tmp.c_address, data);
            fwrite(&(tmp.c_address), sizeof(tmp.c_address), 1, out[2]);
            break;
           case 3:
            // 处理 c_nationkey 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[3]);
            }
            tmp.c_nationkey = strtol(data, NULL, 10);
            fwrite(&(tmp.c_nationkey),sizeof(int), 1, out[3]);
            break;
           case 4:
            // 处理 c_phone 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.c_phone);
              fwrite(&header,sizeof(struct columnHeader),1,out[4]);
            }
            strcpy(tmp.c_phone, data);
            fwrite(&(tmp.c_phone),sizeof(tmp.c_phone), 1, out[4]);
            break;
           case 5:
            // 处理 c_acctbal 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(float);
              fwrite(&header,sizeof(struct columnHeader),1,out[5]);
            }
            tmp.c_acctbal = strtod(data, NULL);
            fwrite(&(tmp.c_acctbal),sizeof(float), 1, out[5]);
            break;
           case 6:
            // 处理 c_mktsegment 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[6]);
            }
            tmp.c_mktsegment = strtol(data, NULL, 10);
            fwrite(&(tmp.c_mktsegment),sizeof(int), 1, out[6]);
            break;
           case 7:
            // 处理 c_comment 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.c_comment);
              fwrite(&header,sizeof(struct columnHeader),1,out[7]);
            }
            strcpy(tmp.c_comment, data);
            fwrite(&(tmp.c_comment),sizeof(tmp.c_comment), 1, out[7]);
            break;
        }
        count++;
      }
    }
    // 为了防止以"\n"作为行间隔造成最后一列未处理。正常应该以"|\n"作为行间隔。
    if(count == 7){
      if(writeHeader == 1){
        header.blockSize = header.tupleNum * sizeof(tmp.c_comment);
        fwrite(&header,sizeof(struct columnHeader),1,out[7]);
      }
      strncpy(tmp.c_comment,buf+prev,i-prev);
      fwrite(&(tmp.c_comment),sizeof(tmp.c_comment), 1, out[7]);
    }
  }

  for(i=0;i<8;i++){
    fclose(out[i]);
  }
}

// fp路径为"tpch/data/s*/part.tbl.p", outName = "PART"
void part (FILE *fp, char *outName){
  struct part tmp;
  struct columnHeader header;
  FILE * out[9];
  char buf[1024] = {0};
  char data [1024] = {0};
  // tupleNum是源文件中的行数，tupleUnit仅为中间变量，tupleCount是当前block中已处理的tuple数量，tupleRemain是源文件剩余未处理的行数
  long tupleNum = 0, tupleUnit = 0, tupleCount = 0, tupleRemain = 0;
  int i = 0, prev = 0,count=0;

  for(i = 0; i < 9; i++){
    char path[PATH_MAX] = {0};
    sprintf(path,"%s%d",outName,i);
    // 每列一个out
    out[i] = fopen(path, "w");
    if(!out[i]){
      printf("Failed to open %s\n",path);
      exit(-1);
    }
  }

  while(fgets(buf,sizeof(buf),fp) !=NULL)
    tupleNum ++;
  header.totalTupleNum = tupleNum;
  tupleRemain = tupleNum;
  if(tupleNum > BLOCKNUM)
    tupleUnit = BLOCKNUM;
  else
    tupleUnit = tupleNum;
  // 当前block的tuple数量
  header.tupleNum = tupleUnit;
  header.format = UNCOMPRESSED;
  header.blockId = 0;
  // block的总数
  header.blockTotal = (tupleNum + BLOCKNUM -1) / BLOCKNUM ;
  fseek(fp,0,SEEK_SET);

  while(fgets(buf,sizeof(buf),fp)!= NULL){
    int writeHeader = 0;
    tupleCount ++;
    // 超过BLOCKNUM，该开始下一个block
    if(tupleCount > BLOCKNUM){
      tupleCount = 1;
      tupleRemain -= BLOCKNUM;
      if (tupleRemain > BLOCKNUM)
        tupleUnit = BLOCKNUM;
      else
        tupleUnit = tupleRemain;
      header.tupleNum = tupleUnit;
      header.blockId ++;
      writeHeader = 1;
    }
    for(i = 0, prev = 0, count = 0; buf[i] !='\n'; i++){
      if (buf[i] == delimiter){
        memset(data,0,sizeof(data));
        // 得到某一列的数据
        strncpy(data,buf+prev,i-prev);
        prev = i+1;
        switch(count){
           case 0:
            // 处理 p_partkey 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[0]);
            }
            tmp.p_partkey = strtol(data,NULL,10);
            fwrite(&(tmp.p_partkey),sizeof(int),1,out[0]);
            break;
           case 1:
            // 处理 p_name 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.p_name);
              fwrite(&header, sizeof(struct columnHeader), 1, out[1]);
            }
            strcpy(tmp.p_name, data);
            fwrite(&(tmp.p_name), sizeof(tmp.p_name), 1, out[1]);
            break;
           case 2:
            // 处理 p_mfgr 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header, sizeof(struct columnHeader), 1, out[2]);
            }
            tmp.p_mfgr = strtol(data, NULL, 10);
            fwrite(&(tmp.p_mfgr), sizeof(int), 1, out[2]);
            break;
           case 3:
            // 处理 p_brand 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.p_brand);
              fwrite(&header,sizeof(struct columnHeader),1,out[3]);
            }
            strcpy(tmp.p_brand, data);
            fwrite(&(tmp.p_brand),sizeof(tmp.p_brand), 1, out[3]);
            break;
           case 4:
            // 处理 p_type 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[4]);
            }
            tmp.p_type = strtol(data, NULL, 10);
            fwrite(&(tmp.p_type),sizeof(int), 1, out[4]);
            break;
           case 5:
            // 处理 p_size 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[5]);
            }
            tmp.p_size = strtol(data, NULL, 10);
            fwrite(&(tmp.p_size),sizeof(int), 1, out[5]);
            break;
           case 6:
            // 处理 p_container 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.p_container);
              fwrite(&header,sizeof(struct columnHeader),1,out[6]);
            }
            strcpy(tmp.p_container, data);
            fwrite(&(tmp.p_container),sizeof(tmp.p_container), 1, out[6]);
            break;
           case 7:
            // 处理 p_retailprice 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(float);
              fwrite(&header,sizeof(struct columnHeader),1,out[7]);
            }
            tmp.p_retailprice = strtod(data, NULL);
            fwrite(&(tmp.p_retailprice),sizeof(float), 1, out[7]);
            break;
           case 8:
            // 处理 p_comment 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.p_comment);
              fwrite(&header,sizeof(struct columnHeader),1,out[8]);
            }
            strcpy(tmp.p_comment, data);
            fwrite(&(tmp.p_comment),sizeof(tmp.p_comment), 1, out[8]);
            break;
        }
        count++;
      }
    }
    // 为了防止以"\n"作为行间隔造成最后一列未处理。正常应该以"|\n"作为行间隔。
    if(count == 8){
      if(writeHeader == 1){
        header.blockSize = header.tupleNum * sizeof(tmp.p_comment);
        fwrite(&header,sizeof(struct columnHeader),1,out[8]);
      }
      strncpy(tmp.p_comment,buf+prev,i-prev);
      fwrite(&(tmp.p_comment),sizeof(tmp.p_comment), 1, out[8]);
    }
  }

  for(i=0;i<9;i++){
    fclose(out[i]);
  }
}

// fp路径为"tpch/data/s*/partsupp.tbl.p", outName = "PARTSUPP"
void partsupp (FILE *fp, char *outName){
  struct partsupp tmp;
  struct columnHeader header;
  FILE * out[5];
  char buf[1024] = {0};
  char data [1024] = {0};
  // tupleNum是源文件中的行数，tupleUnit仅为中间变量，tupleCount是当前block中已处理的tuple数量，tupleRemain是源文件剩余未处理的行数
  long tupleNum = 0, tupleUnit = 0, tupleCount = 0, tupleRemain = 0;
  int i = 0, prev = 0,count=0;

  for(i = 0; i < 5; i++){
    char path[PATH_MAX] = {0};
    sprintf(path,"%s%d",outName,i);
    // 每列一个out
    out[i] = fopen(path, "w");
    if(!out[i]){
      printf("Failed to open %s\n",path);
      exit(-1);
    }
  }

  while(fgets(buf,sizeof(buf),fp) !=NULL)
    tupleNum ++;
  header.totalTupleNum = tupleNum;
  tupleRemain = tupleNum;
  if(tupleNum > BLOCKNUM)
    tupleUnit = BLOCKNUM;
  else
    tupleUnit = tupleNum;
  // 当前block的tuple数量
  header.tupleNum = tupleUnit;
  header.format = UNCOMPRESSED;
  header.blockId = 0;
  // block的总数
  header.blockTotal = (tupleNum + BLOCKNUM -1) / BLOCKNUM ;
  fseek(fp,0,SEEK_SET);

  while(fgets(buf,sizeof(buf),fp)!= NULL){
    int writeHeader = 0;
    tupleCount ++;
    // 超过BLOCKNUM，该开始下一个block
    if(tupleCount > BLOCKNUM){
      tupleCount = 1;
      tupleRemain -= BLOCKNUM;
      if (tupleRemain > BLOCKNUM)
        tupleUnit = BLOCKNUM;
      else
        tupleUnit = tupleRemain;
      header.tupleNum = tupleUnit;
      header.blockId ++;
      writeHeader = 1;
    }
    for(i = 0, prev = 0, count = 0; buf[i] !='\n'; i++){
      if (buf[i] == delimiter){
        memset(data,0,sizeof(data));
        // 得到某一列的数据
        strncpy(data,buf+prev,i-prev);
        prev = i+1;
        switch(count){
           case 0:
            // 处理 ps_partkey 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[0]);
            }
            tmp.ps_partkey = strtol(data,NULL,10);
            fwrite(&(tmp.ps_partkey),sizeof(int),1,out[0]);
            break;
           case 1:
            // 处理 ps_suppkey 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header, sizeof(struct columnHeader), 1, out[1]);
            }
            tmp.ps_suppkey = strtol(data, NULL, 10);
            fwrite(&(tmp.ps_suppkey), sizeof(int), 1, out[1]);
            break;
           case 2:
            // 处理 ps_availqty 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header, sizeof(struct columnHeader), 1, out[2]);
            }
            tmp.ps_availqty = strtol(data, NULL, 10);
            fwrite(&(tmp.ps_availqty), sizeof(int), 1, out[2]);
            break;
           case 3:
            // 处理 ps_supplycost 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[3]);
            }
            tmp.ps_supplycost = strtol(data, NULL, 10);
            fwrite(&(tmp.ps_supplycost),sizeof(int), 1, out[3]);
            break;
           case 4:
            // 处理 ps_comment 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.ps_comment);
              fwrite(&header,sizeof(struct columnHeader),1,out[4]);
            }
            strcpy(tmp.ps_comment, data);
            fwrite(&(tmp.ps_comment),sizeof(tmp.ps_comment), 1, out[4]);
            break;
        }
        count++;
      }
    }
    // 为了防止以"\n"作为行间隔造成最后一列未处理。正常应该以"|\n"作为行间隔。
    if(count == 4){
      if(writeHeader == 1){
        header.blockSize = header.tupleNum * sizeof(tmp.ps_comment);
        fwrite(&header,sizeof(struct columnHeader),1,out[4]);
      }
      strncpy(tmp.ps_comment,buf+prev,i-prev);
      fwrite(&(tmp.ps_comment),sizeof(tmp.ps_comment), 1, out[4]);
    }
  }

  for(i=0;i<5;i++){
    fclose(out[i]);
  }
}

// fp路径为"tpch/data/s*/supplier.tbl.p", outName = "SUPPLIER"
void supplier (FILE *fp, char *outName){
  struct supplier tmp;
  struct columnHeader header;
  FILE * out[7];
  char buf[1024] = {0};
  char data [1024] = {0};
  // tupleNum是源文件中的行数，tupleUnit仅为中间变量，tupleCount是当前block中已处理的tuple数量，tupleRemain是源文件剩余未处理的行数
  long tupleNum = 0, tupleUnit = 0, tupleCount = 0, tupleRemain = 0;
  int i = 0, prev = 0,count=0;

  for(i = 0; i < 7; i++){
    char path[PATH_MAX] = {0};
    sprintf(path,"%s%d",outName,i);
    // 每列一个out
    out[i] = fopen(path, "w");
    if(!out[i]){
      printf("Failed to open %s\n",path);
      exit(-1);
    }
  }

  while(fgets(buf,sizeof(buf),fp) !=NULL)
    tupleNum ++;
  header.totalTupleNum = tupleNum;
  tupleRemain = tupleNum;
  if(tupleNum > BLOCKNUM)
    tupleUnit = BLOCKNUM;
  else
    tupleUnit = tupleNum;
  // 当前block的tuple数量
  header.tupleNum = tupleUnit;
  header.format = UNCOMPRESSED;
  header.blockId = 0;
  // block的总数
  header.blockTotal = (tupleNum + BLOCKNUM -1) / BLOCKNUM ;
  fseek(fp,0,SEEK_SET);

  while(fgets(buf,sizeof(buf),fp)!= NULL){
    int writeHeader = 0;
    tupleCount ++;
    // 超过BLOCKNUM，该开始下一个block
    if(tupleCount > BLOCKNUM){
      tupleCount = 1;
      tupleRemain -= BLOCKNUM;
      if (tupleRemain > BLOCKNUM)
        tupleUnit = BLOCKNUM;
      else
        tupleUnit = tupleRemain;
      header.tupleNum = tupleUnit;
      header.blockId ++;
      writeHeader = 1;
    }
    for(i = 0, prev = 0, count = 0; buf[i] !='\n'; i++){
      if (buf[i] == delimiter){
        memset(data,0,sizeof(data));
        // 得到某一列的数据
        strncpy(data,buf+prev,i-prev);
        prev = i+1;
        switch(count){
           case 0:
            // 处理 s_suppkey 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[0]);
            }
            tmp.s_suppkey = strtol(data,NULL,10);
            fwrite(&(tmp.s_suppkey),sizeof(int),1,out[0]);
            break;
           case 1:
            // 处理 s_name 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header, sizeof(struct columnHeader), 1, out[1]);
            }
            tmp.s_name = strtol(data, NULL, 10);
            fwrite(&(tmp.s_name), sizeof(int), 1, out[1]);
            break;
           case 2:
            // 处理 s_address 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.s_address);
              fwrite(&header, sizeof(struct columnHeader), 1, out[2]);
            }
            strcpy(tmp.s_address, data);
            fwrite(&(tmp.s_address), sizeof(tmp.s_address), 1, out[2]);
            break;
           case 3:
            // 处理 s_nationkey 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[3]);
            }
            tmp.s_nationkey = strtol(data, NULL, 10);
            fwrite(&(tmp.s_nationkey),sizeof(int), 1, out[3]);
            break;
           case 4:
            // 处理 s_phone 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.s_phone);
              fwrite(&header,sizeof(struct columnHeader),1,out[4]);
            }
            strcpy(tmp.s_phone, data);
            fwrite(&(tmp.s_phone),sizeof(tmp.s_phone), 1, out[4]);
            break;
           case 5:
            // 处理 s_acctbal 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(float);
              fwrite(&header,sizeof(struct columnHeader),1,out[5]);
            }
            tmp.s_acctbal = strtod(data, NULL);
            fwrite(&(tmp.s_acctbal),sizeof(float), 1, out[5]);
            break;
           case 6:
            // 处理 s_comment 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.s_comment);
              fwrite(&header,sizeof(struct columnHeader),1,out[6]);
            }
            strcpy(tmp.s_comment, data);
            fwrite(&(tmp.s_comment),sizeof(tmp.s_comment), 1, out[6]);
            break;
        }
        count++;
      }
    }
    // 为了防止以"\n"作为行间隔造成最后一列未处理。正常应该以"|\n"作为行间隔。
    if(count == 6){
      if(writeHeader == 1){
        header.blockSize = header.tupleNum * sizeof(tmp.s_comment);
        fwrite(&header,sizeof(struct columnHeader),1,out[6]);
      }
      strncpy(tmp.s_comment,buf+prev,i-prev);
      fwrite(&(tmp.s_comment),sizeof(tmp.s_comment), 1, out[6]);
    }
  }

  for(i=0;i<7;i++){
    fclose(out[i]);
  }
}

// fp路径为"tpch/data/s*/nation.tbl.p", outName = "NATION"
void nation (FILE *fp, char *outName){
  struct nation tmp;
  struct columnHeader header;
  FILE * out[4];
  char buf[1024] = {0};
  char data [1024] = {0};
  // tupleNum是源文件中的行数，tupleUnit仅为中间变量，tupleCount是当前block中已处理的tuple数量，tupleRemain是源文件剩余未处理的行数
  long tupleNum = 0, tupleUnit = 0, tupleCount = 0, tupleRemain = 0;
  int i = 0, prev = 0,count=0;

  for(i = 0; i < 4; i++){
    char path[PATH_MAX] = {0};
    sprintf(path,"%s%d",outName,i);
    // 每列一个out
    out[i] = fopen(path, "w");
    if(!out[i]){
      printf("Failed to open %s\n",path);
      exit(-1);
    }
  }

  while(fgets(buf,sizeof(buf),fp) !=NULL)
    tupleNum ++;
  header.totalTupleNum = tupleNum;
  tupleRemain = tupleNum;
  if(tupleNum > BLOCKNUM)
    tupleUnit = BLOCKNUM;
  else
    tupleUnit = tupleNum;
  // 当前block的tuple数量
  header.tupleNum = tupleUnit;
  header.format = UNCOMPRESSED;
  header.blockId = 0;
  // block的总数
  header.blockTotal = (tupleNum + BLOCKNUM -1) / BLOCKNUM ;
  fseek(fp,0,SEEK_SET);

  while(fgets(buf,sizeof(buf),fp)!= NULL){
    int writeHeader = 0;
    tupleCount ++;
    // 超过BLOCKNUM，该开始下一个block
    if(tupleCount > BLOCKNUM){
      tupleCount = 1;
      tupleRemain -= BLOCKNUM;
      if (tupleRemain > BLOCKNUM)
        tupleUnit = BLOCKNUM;
      else
        tupleUnit = tupleRemain;
      header.tupleNum = tupleUnit;
      header.blockId ++;
      writeHeader = 1;
    }
    for(i = 0, prev = 0, count = 0; buf[i] !='\n'; i++){
      if (buf[i] == delimiter){
        memset(data,0,sizeof(data));
        // 得到某一列的数据
        strncpy(data,buf+prev,i-prev);
        prev = i+1;
        switch(count){
           case 0:
            // 处理 n_nationkey 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[0]);
            }
            tmp.n_nationkey = strtol(data,NULL,10);
            fwrite(&(tmp.n_nationkey),sizeof(int),1,out[0]);
            break;
           case 1:
            // 处理 n_name 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header, sizeof(struct columnHeader), 1, out[1]);
            }
            tmp.n_name = strtol(data, NULL, 10);
            fwrite(&(tmp.n_name), sizeof(int), 1, out[1]);
            break;
           case 2:
            // 处理 n_regionkey 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header, sizeof(struct columnHeader), 1, out[2]);
            }
            tmp.n_regionkey = strtol(data, NULL, 10);
            fwrite(&(tmp.n_regionkey), sizeof(int), 1, out[2]);
            break;
           case 3:
            // 处理 n_comment 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.n_comment);
              fwrite(&header,sizeof(struct columnHeader),1,out[3]);
            }
            strcpy(tmp.n_comment, data);
            fwrite(&(tmp.n_comment),sizeof(tmp.n_comment), 1, out[3]);
            break;
        }
        count++;
      }
    }
    // 为了防止以"\n"作为行间隔造成最后一列未处理。正常应该以"|\n"作为行间隔。
    if(count == 3){
      if(writeHeader == 1){
        header.blockSize = header.tupleNum * sizeof(tmp.n_comment);
        fwrite(&header,sizeof(struct columnHeader),1,out[3]);
      }
      strncpy(tmp.n_comment,buf+prev,i-prev);
      fwrite(&(tmp.n_comment),sizeof(tmp.n_comment), 1, out[3]);
    }
  }

  for(i=0;i<4;i++){
    fclose(out[i]);
  }
}

// fp路径为"tpch/data/s*/region.tbl.p", outName = "REGION"
void region (FILE *fp, char *outName){
  struct region tmp;
  struct columnHeader header;
  FILE * out[3];
  char buf[1024] = {0};
  char data [1024] = {0};
  // tupleNum是源文件中的行数，tupleUnit仅为中间变量，tupleCount是当前block中已处理的tuple数量，tupleRemain是源文件剩余未处理的行数
  long tupleNum = 0, tupleUnit = 0, tupleCount = 0, tupleRemain = 0;
  int i = 0, prev = 0,count=0;

  for(i = 0; i < 3; i++){
    char path[PATH_MAX] = {0};
    sprintf(path,"%s%d",outName,i);
    // 每列一个out
    out[i] = fopen(path, "w");
    if(!out[i]){
      printf("Failed to open %s\n",path);
      exit(-1);
    }
  }

  while(fgets(buf,sizeof(buf),fp) !=NULL)
    tupleNum ++;
  header.totalTupleNum = tupleNum;
  tupleRemain = tupleNum;
  if(tupleNum > BLOCKNUM)
    tupleUnit = BLOCKNUM;
  else
    tupleUnit = tupleNum;
  // 当前block的tuple数量
  header.tupleNum = tupleUnit;
  header.format = UNCOMPRESSED;
  header.blockId = 0;
  // block的总数
  header.blockTotal = (tupleNum + BLOCKNUM -1) / BLOCKNUM ;
  fseek(fp,0,SEEK_SET);

  while(fgets(buf,sizeof(buf),fp)!= NULL){
    int writeHeader = 0;
    tupleCount ++;
    // 超过BLOCKNUM，该开始下一个block
    if(tupleCount > BLOCKNUM){
      tupleCount = 1;
      tupleRemain -= BLOCKNUM;
      if (tupleRemain > BLOCKNUM)
        tupleUnit = BLOCKNUM;
      else
        tupleUnit = tupleRemain;
      header.tupleNum = tupleUnit;
      header.blockId ++;
      writeHeader = 1;
    }
    for(i = 0, prev = 0, count = 0; buf[i] !='\n'; i++){
      if (buf[i] == delimiter){
        memset(data,0,sizeof(data));
        // 得到某一列的数据
        strncpy(data,buf+prev,i-prev);
        prev = i+1;
        switch(count){
           case 0:
            // 处理 r_regionkey 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header,sizeof(struct columnHeader),1,out[0]);
            }
            tmp.r_regionkey = strtol(data,NULL,10);
            fwrite(&(tmp.r_regionkey),sizeof(int),1,out[0]);
            break;
           case 1:
            // 处理 r_name 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(int);
              fwrite(&header, sizeof(struct columnHeader), 1, out[1]);
            }
            tmp.r_name = strtol(data, NULL, 10);
            fwrite(&(tmp.r_name), sizeof(int), 1, out[1]);
            break;
           case 2:
            // 处理 l_suppkey 列
            if(writeHeader == 1){
              header.blockSize = header.tupleNum * sizeof(tmp.r_comment);
              fwrite(&header, sizeof(struct columnHeader), 1, out[2]);
            }
            strcpy(tmp.r_comment, data);
            fwrite(&(tmp.r_comment), sizeof(tmp.r_comment), 1, out[2]);
            break;
        }
        count++;
      }
    }
    // 为了防止以"\n"作为行间隔造成最后一列未处理。正常应该以"|\n"作为行间隔。
    if(count == 2){
      if(writeHeader == 1){
        header.blockSize = header.tupleNum * sizeof(tmp.r_comment);
        fwrite(&header,sizeof(struct columnHeader),1,out[2]);
      }
      strncpy(tmp.r_comment,buf+prev,i-prev);
      fwrite(&(tmp.r_comment),sizeof(tmp.r_comment), 1, out[2]);
    }
  }

  for(i=0;i<3;i++){
    fclose(out[i]);
  }
}

int main(int argc, char ** argv){
  FILE * in = NULL, *out = NULL;
  int table;
  int setPath = 0;
  char path[PATH_MAX];
  char cwd[PATH_MAX];

  int long_index;
  struct option long_options[] = {
    {"lineitem",required_argument,0,'0'},
    {"orders",required_argument,0,'1'},
    {"customer",required_argument,0,'2'},
    {"part",required_argument,0,'3'},
    {"partsupp",required_argument,0,'4'},
    {"supplier",required_argument,0,'5'},
    {"nation",required_argument,0,'6'},
    {"region",required_argument,0,'7'},
    {"delimiter",required_argument,0,'8'},
    {"datadir",required_argument,0,'9'}
  };

  while((table=getopt_long(argc,argv,"",long_options,&long_index))!=-1){
    switch(table){
      case '9':
        setPath = 1;
        // path = "../data/s*_columnar/"
        strcpy(path,optarg);
        break;
    }
  }

  optind=1;

  // cwd = "tpch/loader/"
  getcwd(cwd,PATH_MAX);
  while((table=getopt_long(argc,argv,"",long_options,&long_index))!=-1){
    switch(table){
      case '0':
        // optarg = "tpch/data/s*/lineitem.tbl.p"
        in = fopen(optarg,"r");
        if(!in){
          printf("Failed to open %s\n",optarg);
          exit(-1);
        }
        if (setPath == 1){
          chdir(path);
        }
        lineitem(in,"LINEITEM");
        if (setPath == 1){
          chdir(cwd);
        }
        fclose(in);
        break;
      case '1':
        // optarg = "tpch/data/s*/orders.tbl.p"
        in = fopen(optarg,"r");
        if(!in){
          printf("Failed to open %s\n",optarg);
          exit(-1);
        }
        if (setPath == 1){
          chdir(path);
        }
        orders(in,"ORDERS");
        if (setPath == 1){
          chdir(cwd);
        }
        fclose(in);
        break;
      case '2':
        // optarg = "tpch/data/s*/customer.tbl.p"
        in = fopen(optarg,"r");
        if(!in){
          printf("Failed to open %s\n",optarg);
          exit(-1);
        }
        if (setPath == 1){
          chdir(path);
        }
        customer(in,"CUSTOMER");
        if (setPath == 1){
          chdir(cwd);
        }
        fclose(in);
        break;
      case '3':
        // optarg = "tpch/data/s*/part.tbl.p"
        in = fopen(optarg,"r");
        if(!in){
          printf("Failed to open %s\n",optarg);
          exit(-1);
        }
        if (setPath == 1){
          chdir(path);
        }
        part(in,"PART");
        if (setPath == 1){
          chdir(cwd);
        }
        fclose(in);
        break;
      case '4':
        // optarg = "tpch/data/s*/partsupp.tbl.p"
        in = fopen(optarg,"r");
        if(!in){
          printf("Failed to open %s\n",optarg);
          exit(-1);
        }
        if (setPath == 1){
          chdir(path);
        }
        partsupp(in,"PARTSUPP");
        if (setPath == 1){
          chdir(cwd);
        }
        fclose(in);
        break;
      case '5':
        // optarg = "tpch/data/s*/supplier.tbl.p"
        in = fopen(optarg,"r");
        if(!in){
          printf("Failed to open %s\n",optarg);
          exit(-1);
        }
        if (setPath == 1){
          chdir(path);
        }
        supplier(in,"SUPPLIER");
        if (setPath == 1){
          chdir(cwd);
        }
        fclose(in);
        break;
      case '6':
        // optarg = "tpch/data/s*/nation.tbl.p"
        in = fopen(optarg,"r");
        if(!in){
          printf("Failed to open %s\n",optarg);
          exit(-1);
        }
        if (setPath == 1){
          chdir(path);
        }
        nation(in,"NATION");
        if (setPath == 1){
          chdir(cwd);
        }
        fclose(in);
        break;
      case '7':
        // optarg = "tpch/data/s*/region.tbl.p"
        in = fopen(optarg,"r");
        if(!in){
          printf("Failed to open %s\n",optarg);
          exit(-1);
        }
        if (setPath == 1){
          chdir(path);
        }
        region(in,"REGION");
        if (setPath == 1){
          chdir(cwd);
        }
        fclose(in);
        break;
      case '8':
        delimiter = optarg[0];
        break;
    }
  }

  return 0;
}