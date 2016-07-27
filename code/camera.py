#!/usr/bin/env python
# encoding=utf-8

from scipy import linalg

class Camera(object):
    """
    相机的类
    """

    def __init__(self, P):
        """
        初始化相机类
        """
        self.P = P
        # 标定矩阵
        self.K = None
        # 旋转矩阵
        self.R = None
        # 平移矩阵
        self.t = None
        # 相机中心
        self.c = None

    def project(self, X):
        """
        :param X: (4, n) 的投影点, 并且对坐标归一化
        :return:
        """
        x = linalg.dot(self.P, X)
        for i in range(3):
            x[i] /= x[2]
        return x