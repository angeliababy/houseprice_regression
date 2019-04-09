# 房价预测回归模型
# part 1 数据预处理

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns  # 基于matplotlib数据可视化库

from scipy import stats
from scipy.stats import norm, skew #for some statistics

# 读取文件（文件中第一列id是标号，train.csv最后一列SalePrice（房价）是预测目标变量，中间是特征变量）
# 训练、测试集
train = pd.read_csv(r'D:\projects\regression\train.csv')
# 预测集
test = pd.read_csv(r'D:\projects\regression\test.csv')

# 第一步，查看数据，前五行
print(train.head(5))
print(test.head(5))

# 保存数据第一行id
train_ID = train['Id']
test_ID = test['Id']
# 去掉id列
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# 1.查看数据大致情况，去掉极端值（离群点）
# 画出GrLivArea（居住面积）与SalePrice（售价）对应的散点图scatter，查看数据大概情况
# 该例子中只查看了GrLivArea，当然还可以查看其它数值列情况
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# 由上图散点图分布，去掉面积大价格低的数据(极端值)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# 2.查看预测样本价格是否符合正态分布并修正(某些模型需要，比如线性回归)
# SalePrice分布图
sns.distplot(train['SalePrice'] , fit=norm)
# SalePrice均值、均方差
(mu, sigma) = norm.fit(train['SalePrice'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# QQ-plot（验证某两组数据是否来自同一分布，离的距离）
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# 由上可知目标变量是右倾斜的，(线性)模型要转化为正态分布的数据
# 使用log(1+x)对SalePrice进行转化
train["SalePrice"] = np.log1p(train["SalePrice"])

# 同上，再次查看目标变量是否达到正态分布
# # SalePrice分布图
sns.distplot(train['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(train['SalePrice'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# 第二步，构造特征工程
# 两文件数据量
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
# 连接两个表，将两个表的数据共同处理
all_data = pd.concat((train, test)).reset_index(drop=True)
# 去掉目标变量SalePrice列
all_data.drop(['SalePrice'], axis=1, inplace=True)
# 生成数据形状大小
print("first all_data size is : {}".format(all_data.shape))

# 1.查看各变量与预测目标相关性，按照右边颜色条进行判断
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()

# 2.缺失值处理
# 查看各变量缺失率及降序排序
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print("missing_data:", missing_data.head(20)) # 列出前20个

# 填补缺失值，None\0\众数\中位数\指定值填充，或者去掉该列
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'):
    all_data[col].fillna("None", inplace=True)
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
'BsmtHalfBath', 'MasVnrArea'):
    all_data[col].fillna(0, inplace=True)
for col in ('MSZoning', "Functional", 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'MSSubClass'):
    all_data[col].fillna(all_data[col].mode()[0], inplace=True)
all_data.drop(['Utilities'], axis=1, inplace=True)
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# 同上，再次检查是否不存在缺失值
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print("missing_data:",missing_data.head())

# 3.构造类别特征
# 数值变为类别
# 将这几个列的属性变为str类型，才可以进行下面编码处理
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

# 将某些特征进行标签编码LabelEncoder0,1,2···，对于回归问题，更多地使用labelEncoding。对于分类问题，更多使用OneHotEncoding。
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))

# 4.组合特征，增加一个重要特征（可不用该步）
# 组合TotalBsmtSF、1stFlrSF、2ndFlrSF（地下室、一楼、二楼面积特征）
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

# 5.倾斜数值特征的变换，skew偏度，用boxcox1p处理（可不用该步）
# 将数值特征提取，即类型不为object
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# 列出数据的偏斜度，降序排序
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print("skew:",skewness.head(10))

# 将偏斜度大于0.75的数值列做一个log转换，使之尽量符合正态分布，因为很多模型的假设数据服从正态分布
skewness = skewness[abs(skewness.Skew) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
print("LabelEncoder：all_data.shape:", all_data.shape)

# 6.将属性特征转化为指示变量，one-hot编码
all_data = pd.get_dummies(all_data)
print("one-hot：all_data.shape:", all_data.shape)

# 数据预处理完成后的总训练集、预测集
train = all_data[:ntrain]
test = all_data[ntrain:]
