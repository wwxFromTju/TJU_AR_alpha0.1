#!/usr/env/bin python
# encoding=utf-8

import pickle
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame, pygame.image
from pygame.locals import *
from scipy import *


width, height = 1000, 747

def set_projection_from_camera(K):
    """
    :param K: 标定好的相机, 即标定矩阵
    :return:
    设置视图
    """

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    fx = K[0, 0]
    fy = K[1, 1]
    fovy = 2 * arctan(0.5 * height / fy) * 180 / pi
    aspect = (width * fy) / (height * fx)

    # 定义远和近的剪切平面
    near = 0.1
    far = 100.0

    # 设置透视
    gluPerspective(fovy, aspect, near, far)
    glViewport(0, 0, width, height)


def set_modelview_from_camera(Rt):
    """
    :param Rt: 相机姿态
    :return:
    从相机姿态中获得模拟视图矩阵
    """

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # 围绕x轴将茶壶旋转90度, 使z轴向上
    Rx = array([[1, 0, 0]
                [0, 0, -1],
                [0, 1, 0]])

    # 获得旋转的最佳逼近
    R = Rt[:, :3]
    U, S, V = linalg.svd(R)
    R = dot(U, V)
    R[0, :] = -R[0, :]

    # 获得平移量
    t = Rt[:, 3]

    # 获得4x4的模拟视图矩阵
    M = eye(4)
    M[:3, :3] = dot(R, Rx)
    M[:3, 3] = t

    # 转置并压平以获得序数值
    M = M.T
    m = M.flatten()

    # 将模拟视图矩阵替换为新的矩阵
    glLoadMatrixf(m)


def draw_background(image):
    """
    :param image: 载入图像获得纹理
    :return:
    使用四边形绘制背景图像
    """

    # 载入背景图像(使用 .bmp格式), 转换为OpenGl纹理
    bg_image = pygame.image.load(image).convert()
    bg_data = pygame.imgae.tostring(bg_image, 'RGBX', 1)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 绑定纹理
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, glGenTextures(1))
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_data)
    # 注意fl与FL
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    # 创建四方形填充整个窗口
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, -1,0)
    glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, -1.0)
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, -1.0)
    glEnd()

    # 清除纹理
    glDeleteTextures(1)


def draw_teapot(size):
    """
    :param size:相对大小为size
    :return:
    在原点处绘制红色茶壶
    """

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)

    # 绘制红色茶壶
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0, 0, 0, 0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.0, 0.0, 0.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.7, 0.6, 0.6, 0.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 0.25 * 128.0)
    glutSolidTeapot(size)


def setup():
    """
    设置窗口和环境
    """

    pygame.init()
    pygame.display.set_mode(((width, height), OPENGL | DOUBLEBUF))
    pygame.display.set_caption('tju ar')

    # 载入
