# -*- coding: utf-8 -*-

#################
# train xgboost #
#################

from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import numpy as np 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer

# load data for training
feature_matrix = np.loadtxt('/media/shao/TOSHIBA EXT/data_object_velodyne/feature_matrix_with_label/train/data/r_0.16.txt')
print('the shape of the loaded feature matrix is ', feature_matrix.shape)
data = feature_matrix[:, :-1]
target = feature_matrix[:, -1]

# xgbc = XGBClassifier(
#     silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
#     #nthread=4,# cpu 线程数 默认最大
#     learning_rate= 0.3, # 如同学习率
#     min_child_weight=1, 
#     # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#     #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#     #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
#     max_depth=6, # 构建树的深度，越大越容易过拟合
#     gamma=0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
#     subsample=1, # 随机采样训练样本 训练实例的子采样比
#     max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
#     colsample_bytree=1, # 生成树时进行的列采样 
#     reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#     #reg_alpha=0, # L1 正则项参数
#     scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
#     objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
#     num_class=5, # 类别数，多分类与 multisoftmax 并用
#     n_estimators=100, #树的个数
#     seed=1000 #随机种子
#     #eval_metric= 'auc'
# )

# xgbc.fit(data, target, eval_metric='auc')
# y_true, y_pred = target, xgbc.predict(data)
# print"Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred)


xgbc = XGBClassifier(   num_class=5,
                        objective='multi:softmax',
                        scale_pos_weight=1, 
                        seed=1000, 
                        colsample_bytree=1,
                        silent=0,
                        subsample=1
                    )
# # coarse tune
# params = {  'learning_rate':[0.1, 0.3, 0.5, 0.7, 0.9],
#             'min_child_weight':[1, 2, 3],
#             'max_depth':[4, 6, 8, 10],
#             'gamma':[0.1, 0.2, 0.3, 0.4],
#             'max_delta_step':[0, 1],
#             'reg_lambda':[1, 1.5, 2, 2.5, 3],
#             'n_estimators':[10, 20, 40, 60, 100, 120]
#         }

# fine tune
params = {  'learning_rate':[0.9, 1.0, 1.1, 1.2, 1.4],  # optimal 1.0
            'min_child_weight':[2],                     # optimal 2
            'max_depth':[8, 9, 10],                     # optimal 9
            'gamma':[0.07, 0.08, 0.09, 0.1],            # optimal 0.09
            'max_delta_step':[0],                       # optimal 0
            'reg_lambda':[3.3, 3.4, 3.5, 3.7, 3.9],     # optimal 3.4
            'n_estimators':[10, 11, 12, 13, 14]         # optimal 13
        }
fone_scorer = make_scorer(fbeta_score, beta=1, average='weighted')
clf = GridSearchCV (
                        xgbc, 
                        params, 
                        scoring=fone_scorer,
                        n_jobs=4, 
                        cv=5, 
                        return_train_score=False, 
                        iid=True
                    )
clf.fit(data, target)

# print important info
print('clf.cv_results_', clf.cv_results_)
print('clf.best_params_', clf.best_params_)
print('clf.best_estimator_', clf.best_estimator_)
print('clf.grid_scores_', clf.grid_scores_) 
print('best score', clf.grid_scores_[clf.best_index_])

# save the trained model
from sklearn.externals import joblib
joblib.dump(clf, 'xgb.pkl')