#!/usr/bin/env python3
import sys

file_name = None 
index = 0
try : 
    file_name = sys.argv[1]
    index = int(sys.argv[2])
    f = open(sys.argv[1])
    # Input from standard input
    for line in f:
        line = line.strip().split("\t")
        cases = str(line[index]).replace(',','')
        print(f"{line[0]} {int(cases)}")
except : 
    pass