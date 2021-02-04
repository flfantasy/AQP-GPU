l_returnflag = "R,A,N".split(",")

l_linestatus = "O,F".split(",")

type1 = "STANDARD,SMALL,MEDIUM,LARGE,ECONOMY,PROMO".split(",")
type2 = "ANODIZED,BURNISHED,PLATED,POLISHED,BRUSHED".split(",")
type3 = "TIN,NICKEL,BRASS,STEEL,COPPER".split(",")
p_type = []
for i in type1:
    for j in type2:
        for k in type3:
            p_type.append(i + " " + j + " " + k)

c_mktsegment = "AUTOMOBILE,BUILDING,FURNITURE,MACHINERY,HOUSEHOLD".split(",")

n_name = "ALGERIA,ARGENTINA,BRAZIL,CANADA,EGYPT,ETHIOPIA,FRANCE,GERMANY,INDIA,INDONESIA,IRAN,IRAQ,JAPAN,JORDAN,KENYA,MOROCCO,MOZAMBIQUE,PERU,CHINA,ROMANIA,SAUDI ARABIA,VIETNAM,RUSSIA,UNITED KINGDOM,UNITED STATES".split(",")

r_name = "AFRICA,AMERICA,ASIA,EUROPE,MIDDLE EAST".split(",")

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'convert')
    parser.add_argument('data_directory', type=str, help='Data Directory')
    args = parser.parse_args()
    data_dir = args.data_directory

    # process lineitem
    lines = open(data_dir + 'lineitem.tbl').readlines()
    o = []
    for line in lines:
        try:
            # 8->l_returnflag, 9->l_linestatus, 10->l_shipdate
            parts = line.split('|')
            parts[8] = str(l_returnflag.index(parts[8]))
            parts[9] = str(l_linestatus.index(parts[9]))
            parts[10] = parts[10].replace("-","")
            o.append('|'.join(parts))
        except:
            print(line)
            break
    f = open(data_dir + 'lineitem.tbl.p','w')
    for line in o:
        f.write(line)
    f.close()

    # process orders
    lines = open(data_dir + 'orders.tbl').readlines()
    o = []
    for line in lines:
        try:
            # 4->o_orderdate
            parts = line.split('|')
            parts[4] = parts[4].replace("-","")
            o.append('|'.join(parts))
        except:
            print(line)
            break

    f = open(data_dir + 'orders.tbl.p','w')
    for line in o:
        f.write(line)
    f.close()

    # process customer
    lines = open(data_dir + 'customer.tbl').readlines()
    o = []
    for line in lines:
        try:
            # 6->c_mktsegment
            parts = line.split('|')
            parts[6] = str(c_mktsegment.index(parts[6]))
            o.append('|'.join(parts))
        except:
            print(line)
            break

    f = open(data_dir + 'customer.tbl.p','w')
    for line in o:
        f.write(line)
    f.close()

    # process part
    lines = open(data_dir + 'part.tbl').readlines()
    o = []
    for line in lines:
        try:
            # 2->p_mfgr, 4->p_type
            parts = line.split('|')
            parts[2] = int(parts[2].split('#')[-1]) - 1
            parts[2] = str(parts[2])
            parts[4] = str(p_type.index(parts[4]))
            o.append('|'.join(parts))
        except:
            print(line)
            break

    f = open(data_dir + 'part.tbl.p','w')
    for line in o:
        f.write(line)
    f.close()

    # process supplier
    lines = open(data_dir + 'supplier.tbl').readlines()
    o = []
    for line in lines:
        try:
            # 1->s_name, 4->s_phone
            parts = line.split('|')
            parts[1] = parts[0]
            parts[4] = parts[4].replace("-","")
            o.append('|'.join(parts))
        except:
            print(line)
            break

    f = open(data_dir + 'supplier.tbl.p','w')
    for line in o:
        f.write(line)
    f.close()

    # process nation
    lines = open(data_dir + 'nation.tbl').readlines()
    o = []
    for line in lines:
        try:
            # 1->n_name
            parts = line.split('|')
            parts[1] = parts[0]
            o.append('|'.join(parts))
        except:
            print(line)
            break

    f = open(data_dir + 'nation.tbl.p','w')
    for line in o:
        f.write(line)
    f.close()

    # process region
    lines = open(data_dir + 'region.tbl').readlines()
    o = []
    for line in lines:
        try:
            # 1->r_name
            parts = line.split('|')
            parts[1] = parts[0]
            o.append('|'.join(parts))
        except:
            print(line)
            break

    f = open(data_dir + 'region.tbl.p','w')
    for line in o:
        f.write(line)
    f.close()
