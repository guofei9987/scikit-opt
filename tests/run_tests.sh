#!/bin/bash

# -*- coding: utf-8 -*-
# @Time    : 2019/10/15
# @Author  : github.com/Agrover112

echo -n "Running tests...."
echo  -e "\n Testing demo funcs"
python3 test_demo_func.py
echo -e "\n Testing x2gray......"
python3 test_x2gray.py
#echo -e "\nTesting xyz......" For future reference
#python3 test_xyz.py
echo "All tests done!"
echo -n -e "----------------------------------------------------------------------\n"