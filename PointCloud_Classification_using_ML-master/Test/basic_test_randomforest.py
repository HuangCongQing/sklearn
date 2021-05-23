'''
Description: 
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-05-23 18:53:02
LastEditTime: 2021-05-23 18:59:38
FilePath: /sklearn/PointCloud_Classification_using_ML-master/Test/basic_test_randomforest.py
'''
# encoding=utf-8

##############################
# basic_test_randomforest.py #
##############################

from sklearn.externals import joblib
import numpy as np  
from sklearn import metrics
import os
import sys

# get model file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = ROOT_DIR + '/Training/rf.pkl'

# load data for testing
feature_matrix = np.loadtxt('/media/shao/TOSHIBA EXT/data_object_velodyne/feature_matrix_with_label/test/r_0.16.txt')
data = feature_matrix[:, :-1]
target = feature_matrix[:, -1]

# load the trained model加载模型文件
rfc = joblib.load(MODEL_DIR)

# prediction / test 得到各种参数  accuracy，confusion，precision，log
y_pred = rfc.predict(data)
score = metrics.accuracy_score(target, y_pred) # 得到score
print('accuracy score = ', score)
conf_matrix = metrics.confusion_matrix(target, y_pred, [0,1,2,3,4])
print('confusion matrix = ')
print(conf_matrix)
recall = metrics.recall_score(target, y_pred, average='weighted')
print('recall score = ', recall)
precision = metrics.precision_score(target, y_pred, average='weighted')
print('precision score = ', precision)
f1 = metrics.f1_score(target, y_pred, average='weighted')
print('f1 score = ', f1)
prob = rfc.predict_proba(data)
log_loss = metrics.log_loss(target, prob, labels=np.array([0,1,2,3,4]))
print('log loss = ', log_loss)