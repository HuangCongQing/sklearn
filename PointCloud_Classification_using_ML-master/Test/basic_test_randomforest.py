'''
Description: 
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-05-23 18:53:02
LastEditTime: 2021-05-24 09:52:50
FilePath: /sklearn/PointCloud_Classification_using_ML-master/Test/basic_test_randomforest.py
'''
# encoding=utf-8

##############################
# basic_test_randomforest.py #
##############################

from sklearn.externals import joblib # sklearn的版本太新了
# import joblib
import numpy as np  
from sklearn import metrics
import os
import sys

# get model file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
# print("ROOT_DIR", ROOT_DIR)
# MODEL_DIR = ROOT_DIR + '/Training/rf.pkl'
MODEL_DIR = "/home/hcq/python/sklearn/PointCloud_Classification_using_ML-master/Training/rf20210523.pkl"

# load data for testing
# feature_matrix = np.loadtxt('/media/shao/TOSHIBA EXT/data_object_velodyne/feature_matrix_with_label/test/r_0.16.txt')
feature_matrix = np.loadtxt('/home/hcq/data/KITTI_Detection/cutObject/train.txt')
data = feature_matrix[:, :-1]
target = feature_matrix[:, -1]

# load the trained model加载模型文件
rfc = joblib.load(MODEL_DIR)

# prediction / test 得到各种参数  accuracy，confusion，precision，log
y_pred = rfc.predict(data)
print("y_pred: ", y_pred)
score = metrics.accuracy_score(target, y_pred) # 得到score
print('accuracy score = ', score) # accuracy score =  0.9999622313706236
conf_matrix = metrics.confusion_matrix(target, y_pred, [0,1,2,3,4])
print('confusion matrix = ')
print(conf_matrix)
recall = metrics.recall_score(target, y_pred, average='weighted')
print('recall score = ', recall)
precision = metrics.precision_score(target, y_pred, average='weighted')
print('precision score = ', precision)
f1 = metrics.f1_score(target, y_pred, average='weighted')
print('f1 score = ', f1)
prob = rfc.predict_proba(data) #    predict_proba返回的是一个 n 行 k 列的数组， 第 i 行 第 j 列上的数值是模型预测 第 i 个预测样本为某个标签的概率，并且每一行的概率和为1。
# log_loss = metrics.log_loss(target, prob, labels=np.array([0,1,2,3,4])) # ValueError: The number of classes in labels is different from that in y_pred. Classes found in labels: [0 1 2 3 4]
log_loss = metrics.log_loss(target, prob) # log loss =  0.004525287636016965
print('log loss = ', log_loss)