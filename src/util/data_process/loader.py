# -*- coding: utf-8 -*-
# @Time    : 2019/9/11 0:10
# @Author  : chaucerhou

import numpy as np


def read_csv(path, encoding="utf-8", dtype=str, skiprows=1, delimiter=","):
    with open(path, encoding=encoding) as f:
        data = np.loadtxt(f, dtype=dtype, delimiter=delimiter, skiprows=skiprows)
    return data
