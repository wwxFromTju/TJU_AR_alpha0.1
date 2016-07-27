#!/usr/bin/env python
# encoding=utf-8

from PIL import Image
from scipy import *
from pylab import *
import numpy as np
import os


def process_image(imagename, resultname, params='--edge-thresh 10 --peak-thresh 5'):
    """
    :param imagename:
    :param resultname:
    :param params:
    :return:
    """

    if imagename[3:] != 'pgm':
        # 创建一个pgm文件
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    cmmd = str('sift ' + imagename + ' --output=' + resultname + ' ' + params)
    os.system(cmmd)
    print 'processed', imagename, 'to', resultname


def read_features_from_file(filename):
    """
    :param filename:
    :return:
    """

    f = np.loadtxt(filename)
    # 特征位置, 描述子
    return f[:, :4], f[:, :4]


def write_features_to_file(filename, locs, desc):
    """
    :param filename:
    :param locs:
    :param desc:
    :return:
    """

    np.savetxt(filename, np.hstack((locs, desc)))


def plot_features(im, locs, circle=False):
    """
    :param im:
    :param locs:
    :param circle:
    :return:
    """

    def draw_circle(c, r):
        t = np.arange(0, 1.01, .01) * 2 * np.pi
        x = r * np.cos(t) + c[0]
        y = r * np.sin(t) + c[1]
        plot(x, y, 'b', linewidth=2)

    imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plot(locs[:, 0], locs[:, 1], 'ob')
    axis('off')


def match(desc1, desc2):
    """
    :param desc1:
    :param desc2:
    :return:
    对于第一篇图像中的每个描述子,选取其在第二幅图像中的匹配
    """

    desc1 = array([d/linalg.norm(d) for d in desc1])
    desc2 = array([d/linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape

    matchscores = zeros((desc1_size[0], 1), 'int')
    # 预先计算转置
    desc2t = desc2.T
    for i in range(desc1_size[0]):
        dotprods = dot(desc1[i, :], desc2t)
        dotpords = 0.9999 * dotprods
        # 反余弦和反排序, 返回第二幅图像中特征的索引
        indx = argsort(arccos(dotprods))

        # 检查最近邻的角度是否小于dist_ratio乘以第二近领的角度
        if arccos(dotprods)[indx[0]] < dist_ratio * arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])

    return matchscores


def match_twosided(desc1, desc2):
    """
    :param desc1:
    :param desc2:
    :return:
    双向对称版本的match()
    """

    matches_12 = match(desc1, desc2)
    matches_21 = match(desc2, desc1)

    ndx_12 = matches_12.nonzero()[0]

    # 去除不对称的匹配
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

    return matches_12
