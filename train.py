# 房价预测回归模型
# part 2训练部分
import numpy as np

import inference

# 第三步，随机生成训练、测试集
from sklearn.model_selection import train_test_split
train = inference.train   # 训练、测试总数据
y_train = inference.y_train # 目标变量
# 利用sklearn的train_test_split，test_size为测试集所占比例，train训练，test测试
X_train, X_test, y_train, y_test = train_test_split(
    train, y_train, random_state=42, test_size=.25)

# 第四步，进入模型
from sklearn.metrics import mean_squared_error  # RMSE均方根误差
def try_different_method(model):
    model.fit(X_train,y_train)  # 拟合模型
    score = model.score(X_test, y_test)  # 模型评分
    result = model.predict(X_test)  # 测试集结果（目标变量）
    # 用expm1反转，因为之前做过正态处理log(1+x)
    # 测试集的均方根误差(真实目标变量与真实预测的目标变量比较)
    print('RMSE is: ', mean_squared_error(np.expm1(y_test), np.expm1(result)))
    # 模型评分
    print('score is: ', score)


###########3.具体模型选择##########
####3.1决策树回归####
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
model_LinearRegression = LinearRegression()
model_RidgeCV = RidgeCV() # L1
model_LassoCV = LassoCV() # L2
model_ElasticNetCV = ElasticNetCV() # L1、L2组合
####3.3SVM回归####
from sklearn import svm
model_SVR = svm.SVR()
####3.4KNN回归####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####3.5随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()

###########4.具体模型调用部分##########
# 修改此处，单个验证每个模型的评分和RMSE
model = model_GradientBoostingRegressor
try_different_method(model)
