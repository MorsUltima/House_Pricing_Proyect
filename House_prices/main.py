#Import
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes._axes import _log as matplotlib_axes_logger


matplotlib_axes_logger.setLevel('ERROR')
from sklearn.linear_model import LinearRegression

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
all_features = pd.concat([df_train, df_test]).reset_index(drop=True)
#View Settings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Visualize
#print(df_train.head(10))

#correlation matrix
#corrmat = df_train.corr()
#f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrmat, vmax=.8, square=True);


#correlation matrix 2
k = 10
corrmat = df_train.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
#sns.heatmap(cm, vmax = 0.8,annot= True,cbar=True,fmt='.2f',annot_kws={'size': 10},
            #square=True,yticklabels=cols.values, xticklabels=cols.values)

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
#sns.pairplot(df_train[cols], size = 2.5)


#Fix Data

#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

print(df_train.sort_values(by = 'GrLivArea', ascending = False)[:2])
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

#histogram and normal probability plot
df_train['SalePrice'] = np.log(df_train['SalePrice'])


#Griving
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])



#Bs Boss
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0, 'HasBsmt'] = 1
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])



df_train.drop(df_train.columns.difference(['SalePrice', 'OverallQual', 'GrLivArea',
                                           'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']), 1, inplace=True)
df_test.drop(df_test.columns.difference(['Id','SalePrice', 'OverallQual', 'GrLivArea',
                                           'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']), 1, inplace=True)
y = df_train.SalePrice
plt.show()
#Machine learning
X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size = 0.3, random_state = 0)

model = ElasticNetCV()
model.fit(X_train,y_train)
Y_pred = model.predict(X_test)
model.score(X_train,y_train)
acc_ElasticNetCV = round(model.score(X_train, y_train) * 100, 2)
print(acc_ElasticNetCV)
print(df_test.head())
#Submision

print('Predict submission')
submission = pd.read_csv("sample_submission.csv")
q1 = submission['SalePrice'].quantile(0.005)
q2 = submission['SalePrice'].quantile(0.995)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission.csv", index=False)