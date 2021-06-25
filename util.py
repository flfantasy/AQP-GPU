import argparse
import os
import time
import sys

def compile(query_num):
    print('make bin/ssb/q' + query_num + '_hyper_bootstrap')
    os.system('make bin/ssb/q' + query_num + '_hyper_bootstrap 1>>output.log 2>&1')


def run(query_num):
    print('./bin/ssb/q' + query_num + '_hyper_bootstrap')
    os.system('./bin/ssb/q' + query_num + '_hyper_bootstrap 1>>output.log 2>&1')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='compile and run bootstrap on CPU(hyper) or on GPU(crystal)).')
    parser.add_argument('-q', metavar='query_num', nargs='+', help='choose a query to compile and run')
    parser.add_argument('-a', action='store_true', help='compile and run all query')
    parser.add_argument('model', type=str, choices=['cpu', 'gpu'], help='choose running model')
    args = parser.parse_args()

    query_nums = [\
        '11', '12', '13',\
        '21', '22', '23',\
        '31', '32', '33', '34',\
        '41', '42', '43']

    start = time.time();
    if args.a :
        for i in query_nums:
            compile(i)
        for i in query_nums:
            run(i)
    elif args.q != None :
        for i in args.q:
            compile(i)
        for i in args.q:
            run(i)
    else :
        print('arguments should have "-a" or "-q".')
        sys.exit()
    end = time.time()
    print('\n\ntotal time taken: %fs' % end - start)
