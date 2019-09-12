# -*- coding: utf-8 -*-
# @Time    : 2019/9/11 0:31
# @Author  : chaucerhou
from src.util.data_process import loader

if __name__ == '__main__':
    root_path = "../../dataset/digit-recoginzer"
    train_data = loader.read_csv(root_path + "/train.csv")
    test_data = loader.read_csv(root_path + "/test.csv")
    