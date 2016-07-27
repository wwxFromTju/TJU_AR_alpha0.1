#!/usr/bin/env python
# encoding=utf-8

import homography
import camera
import sift
from scipy import *
from pylab import *
from PIL import Image




def my_calibration(sz):
    """
    :param sz: 设置的显示大小
    :return:

    相机参数:
    fx=1229
    cx=360
    fy=1153
    cy=640

    图像大小:
    3264 × 2448
    """

    row, col = sz
    fx = 1153 * col / 3264
    fy = 1229 * row / 2448
    K = diag([fx, fy, 1])
    K[0, 2] = 0.5 * col
    K[1, 2] = 0.5 * row
    return K


def cube_points(c, wid):
    """
    :param c:
    :param wid:
    :return:
    """
    p = []
    # 底部
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])

    #顶部
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])

    # 竖直边
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])

    return array(p).T


# 计算特征
sift.process_image('book_main.jpg', 'im0.sift')
l0, d0 = sift.read_features_from_file('im0.sift')


imname = 'book_main.jpg'
im1 = array(Image.open(imname).convert('L'))
l1, d1 = sift.read_features_from_file('im0.sift')

figure()
gray()
sift.plot_features(im1, l1, circle=True)
show()




sift.process_image('book_test.jpg', 'im1.sift')
l1, d1 = sift.read_features_from_file('im1.sift')


imname = 'book_test.jpg'
im1 = array(Image.open(imname).convert('L'))
l1, d1 = sift.read_features_from_file('im1.sift')

figure()
gray()
sift.plot_features(im1, l1, circle=True)
show()





# 匹配特征
matches = sift.match_twosided(d0, d1)
ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2, :2].T)

print 'fp', fp
print 'tp', tp

model = homography.RansacModel()
H = homography.H_from_ransac(fp, tp, model)

# 计算相机标定矩阵
K = my_calibration((747, 1000))

# 位于边长为0.2, z=0平面上的三维点
box = cube_points([0, 0, 0.1], 0.1)

# 投影第一幅图像上底部的正方形
cam1 = camera.Camera(hstack((K, dot(K, array([[0], [0], [-1]])))))
# 底部正方形上的点
box_cam1 = cam1.project(homography.make_homog(box[:, :5]))

# 使用H将点变换到第二幅图像上
box_trans = homography.normalize(dot(H, box_cam1))

# 从cam1和H中计算第二个照相机矩阵
cam2 = camera.Camera(dot(H, cam1.P))
A = dot(linalg.inv(K), cam2.P[:, :3])
A = array([A[:, 0], A[:, 1], cross(A[:, 0], A[:, 1])]).T
cam2.P[:, :3] = dot(K, A)

# 使用第二个照相机矩阵投影
box_cam2 = cam2.project(homography.make_homog(box))

# 测试: 将点投影在z=0上, 应该能够得到相同的点
point = array([1, 1, 0, 1]).T
print homography.normalize(dot(dot(H, cam1.P), point))
print cam2.project(point)
