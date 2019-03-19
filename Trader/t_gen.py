import csv
import glob
import pandas as pd
import numpy as np


# Function to find the number closest
# to n and divisible by m
def closestNumber(n, m):
    # Find the quotient
    q = int(n / m)

    # 1st possible closest number
    n1 = m * q

    # 2nd possible closest number
    if ((n * m) > 0):
        n2 = (m * (q + 1))
    else:
        n2 = (m * (q - 1))

        # if true, then n1 is the required closest number
    if (abs(n - n1) < abs(n - n2)):
        return n1

        # else n2 is the required closest number
    return n2


def generate_test(infile):
    trainfile = 'train_data/' + infile + '.csv'
    testfile = 'test_data/' + infile + '.csv'
    infile = 'data/' + infile + '.csv'

    size = sum(1 for line in open(infile))

    # Open both files
    with open(infile,'r') as f_in, open(trainfile,'w') as train_out, open(testfile, 'w') as test_out:
        # Write header unchanged
        header = f_in.readline()
        #train_out.write(header)
        #test_out.write(header)

        if (size*0.1) < 100:
            n = 100
        else:
            #calculate 10% of total rows if 10% is less than 100 rows
            n = closestNumber(int(size*0.1),375)

        # n = size - n      // need the latest 100 for test for last 100
        counter = 1
        # seperate into 2 files
        for line in f_in:
            if counter > n:
                train_out.write(line)        #.replace('"','')
            else:
                test_out.write(line)         #.replace('"','')

            counter += 1


    return 0
#----------------------------main--------------------------
abc = glob.glob(".\data\*.csv")
data_dir = []

for f in abc:
    data_dir.append(f.lstrip(".\\data\\").rstrip(".csv"))

for in_file in data_dir:
    print("Processing :",in_file)
    generate_test(in_file)