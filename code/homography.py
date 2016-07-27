#!/usr/bin/env python
# encoding=utf-8
from scipy import *
import numpy as np


def normalize(points):
    """
    :param points:
    :return:
    在齐次坐标意义下, 对点集进行归一化, 使得最后一行为1
    """
    for row in points:
        row /= points[-1]
    return points


def make_homog(points):
    """
    :param points:
    :return:
    """
    return vstack((points, ones((1, points.shape[1]))))


def H_from_points(fp, tp):
    """
    :param fp:
    :param tp:
    :return:
    使用线性DLT方法, 计算单应性矩阵H, 使fp映射到tp。点自动进行归一化
    """

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # 对点进行归一化
    #  ---映射起始点---
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = dot(C1, fp)

    # ---映射对应点---
    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9
    C2 = diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp = dot(C2, tp)

    # 创建用于线性方法的矩阵, 对于每个对应对, 在矩阵中会出现两行数值
    nbr_correspondences = fp.shape[1]
    A = zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0,
                    tp[0][i] * fp[0][i], tp[0][i] * fp[1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1,
                        tp[1][i] * fp[0][i], tp[1][i] * fp[1][i], tp[1][i]]

    U, S, V = np.linalg.svd(A)
    H = V[8].reshape((3, 3))

    # 反归一化
    H = dot(np.linalg.inv(C2), dot(H, C1))

    # 归一化
    return H / H[2, 2]


class RansacModel(object):
    """
    """

    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        """
        :param data:
        :return:
        计算选取的4个对应的单应性矩阵
        """
        # 将其转置, 来调用H_from_points()计算单应性矩阵
        data = data.T

        # 映射的起始点
        fp = data[:3, :4]
        # 映射的目标点
        tp = data[3:, :4]
        # 计算单应性矩阵, 然后返回
        return H_from_points(fp, tp)

    def get_error(self, data, H):
        data = data.T

        fp = data[:3]
        tp = data[3:]

        fp_transformed = dot(H, fp)

        for i in range(3):
            fp_transformed[i] /= fp_transformed[2]

        # 返回每个点的误差
        return sqrt(sum((tp - fp_transformed) ** 2, axis=0))


def H_from_ransac(fp, tp, model, maxiter=1000, match_theshold=10):
    import ransac
    data = vstack((fp, tp))
    H, ransac_data = ransac.ransac(data.T, model, 4, maxiter, match_theshold, 10,
                                   return_all=True)
    return H, ransac_data['inliers']
