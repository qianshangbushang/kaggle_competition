# -*- coding: utf-8 -*-
# @Time    : 2019/9/11 0:16
# @Author  : chaucerhou
from util.data_process import loader

if __name__ == '__main__':
    file_path = ""

    data = loader.read_csv("../dataset/test/test.csv", skiprows=0)
    print(data[1, 0:4])
    print(data.dtype)
