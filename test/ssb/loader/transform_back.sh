cd /home/zhaoh/crystal/test/ssb/loader/
vim ../data/s1/customer.tbl.p << EOF
:%s/$/|/
:wq
EOF
vim ../data/s1/date.tbl << EOF
:%s/$/|/
:wq
EOF
vim ../data/s1/lineorder.tbl << EOF
:%s/$/|/
:wq
EOF
vim ../data/s1/part.tbl.p << EOF
:%s/$/|/
:wq
EOF
vim ../data/s1/supplier.tbl.p << EOF
:%s/$/|/
:wq
EOF
