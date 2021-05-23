# -*- coding: utf-8 -*-
#######################################
########## Data Augmentation ##########
#######################################

import numpy as np

###########
# 绕Z轴旋转 #
###########
# point: vector(1*3:x,y,z)
# rotation_angle: scaler 0~2*pi
def rotate_point (point, rotation_angle):
    point = np.array(point)
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, sin_theta, 0],
                                [-sin_theta, cos_theta, 0],
                                [0, 0, 1]])
    rotated_point = np.dot(point.reshape(-1, 3), rotation_matrix)
    return rotated_point

# point = np.array([1,2,3])
# rotated_point = rotate_point(point, 0.1*np.pi)
# print rotated_point


###########
# 在XYZ上加高斯噪声 #
###########
def jitter_point(point, sigma=0.01, clip=0.05):
    assert(clip > 0)
    point = np.array(point)
    point = point.reshape(-1,3)
    Row, Col = point.shape
    jittered_point = np.clip(sigma * np.random.randn(Row, Col), -1*clip, clip)
    jittered_point += point
    return jittered_point


# jittered_point = jitter_point(point)
# print jittered_point


###########
# Data Augmentation #
###########
def augment_data(point, rotation_angle, sigma, clip):
    return jitter_point(rotate_point(point, rotation_angle), sigma, clip)


point = [0,1,3]
print augment_data(point, 3.14, 0.01, 0.05)

