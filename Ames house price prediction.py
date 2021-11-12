#!/usr/bin/env python
# coding: utf-8

# Import libraries

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
import scipy.stats as stats
from scipy.stats import skew,norm
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import copy
from datetime import datetime
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import sklearn.linear_model as linear_model
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso 
from sklearn.linear_model import BayesianRidge 
from sklearn.linear_model import ElasticNet 
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import r2_score as r2 
sns.set_style('whitegrid') 
plt.rcParams['figure.figsize'] = (20, 10) 

# load and view dataset 

train_df=pd.read_csv('C:/Users/1/Desktop/Machine learning project/train.csv')

test_df=pd.read_csv('C:/Users/1/Desktop/Machine learning project/test.csv')

train_df.info()

train_df.shape

train_df.head()

# Split data into Categorical and numeric features

train_df.describe(include=['object'])

train_df.describe(include=['int64'])

# Spliting Target variable

target=train_df['SalePrice']
target.head()

# Plot distribution of Target variable

import seaborn as sns
sns.distplot(target, hist=True)

# log transformation

# Transform Target variable

target_log=np.log(target)

sns.distplot(target_log, hist=True)

# Compare transformed target variable vs untransformed

matplotlib.rcParams['figure.figsize']=(12.0, 6.0)
prices = pd.DataFrame({"Sale Price": train_df["SalePrice"], "Log SalePrice": target_log})
prices.hist()

train_raw=train_df

train_raw.head()

# Drop target variable from dataset

train_df1=train_df.drop(["SalePrice"], axis=1)

train_df1.head()

# Convert Year sold and Month sold variables to string


train_df1['YrSold']=train_df1['YrSold'].astype(str)

train_df1['MoSold']=train_df1['MoSold'].astype(str)

# Drop TotalBsmtSF, 1stFlrSF, 2ndFlrSF, ID variables


train_df1=train_df1.drop(["TotalBsmtSF"], axis=1)
train_df1=train_df1.drop(["1stFlrSF"], axis=1)
train_df1=train_df1.drop(["2ndFlrSF"], axis=1)
train_df1=train_df1.drop(["Id"], axis=1)

train_df1.head()

# Seperate Categorical and numeric variables

Categorical_columns=[col for col in train_df1.columns.values if train_df1[col].dtype=='object']

train_cat=train_df1[Categorical_columns]

train_num=train_df1.drop(Categorical_columns, axis=1)


train_num.describe()

train_cat.head()
# Plot numeric variables

train_num.hist(figsize=(16,20), bins=50, xlabelsize=8, ylabelsize=8)
#  Skew variables

from scipy.stats import skew

train_num_skew=train_num.apply(lambda x:skew(x.dropna()))
train_num_skew=train_num_skew[train_num_skew>0.75]


train_num[train_num_skew.index]=np.log1p(train_num[train_num_skew.index])



train_num_skew

# # Plot of skewed variables

train_num.hist(figsize=(16,20), bins=50, xlabelsize=8, ylabelsize=8)

# Summary of numeric variables

train_num=((train_num-train_num.mean())/(train_num.max()-train_num.mean()))

train_num.describe()

# Plot of normalized variables

train_num.hist(figsize=(16,20), xlabelsize=8, ylabelsize=8)

# Bar chart of missing values

houseprice_null=train_df1.isnull().sum()

houseprice_null=train_df1.isnull().sum()
houseprice_null=houseprice_null[houseprice_null>0]
houseprice_null.sort_values(inplace=True)
houseprice_null.plot.bar()

# Count and percentage of missing values

missing_total=train_df1.isnull().sum().sort_values(ascending=False)
missing_percent=(train_df1.isnull().sum()/train_df1.isnull().count().sort_values(ascending=False))
missing_data=pd.concat([missing_total ,missing_percent], axis=1, keys=['missing_total', 'missing_percent'])
missing_data.head(15)

sns.set_style("whitegrid")
missing_data.plot.bar()

# handling missing values in numeric features

train_num_len=train_num.shape[0]
print(train_num_len)

for col in train_num.columns.values:
    missing_values=train_num[col].isnull().sum()
    if missing_values>260:
        train_num=train_num(col, axis=1)
    else:
        train_num=train_num.fillna(train_num[col].median())
        
# handling missing values in categorical features

train_cat_len=train_cat.shape[0]
print(train_cat_len)

# Dropping columns with most missing values

for col in train_cat.columns.values:
    missing_values=train_cat[col].isnull().sum()
    if missing_values>50:
        print("dropping column:()", format (col))
        train_cat.drop(col,axis=1)
    else:
         pass                                  

train_cat.describe()

# Dummy Coding of categorical features

train_cat.columns

# Convert categorical variables to dummy

train_cat_dummies=pd.get_dummies(train_cat, drop_first=True)

train_cat_dummies.head(0)

train_cat_dummies.head(9)

# Print total number of categorical and numeric variables


print("numerical features:" + str(len(train_num.columns)))
print("categorical features:" + str(len(train_cat_dummies.columns)))

# Combine cleaned numeric and categorical variables into a single Dataframe

final_data=pd.concat([train_num, train_cat_dummies], axis=1)

#  Factor plot of FireplaceQu with Sale Price

sns.factorplot("Fireplaces", "SalePrice", data=train_df, hue="FireplaceQu")

# Missing Fireplace

FireplaceQu=train_df["FireplaceQu"].fillna('none')
pd.crosstab(train_df.Fireplaces, train_df.FireplaceQu)

# Bar plot of Overall Quality of house with Sale Price

sns.barplot(train_df.OverallQual, train_df.SalePrice)

# MSZoning

# Pie Chart showing Zone house is located

labels=train_df["MSZoning"].unique()
Zone_size=train_df["MSZoning"].value_counts().values
print(Zone_size)

explode=[0.1,0,0,0,0.8]
percent=100.*Zone_size/Zone_size.sum()
labels=['{0}-{1:1.1f} %'.format(i,j) for i,j in zip(labels, percent)]

colors=['black','gold','lightblue','lightcoral','blue']

patches, texts=plt.pie(Zone_size, colors=colors, explode=explode, 
                       shadow=True, startangle=0)
plt.legend(patches,labels,loc="best")
plt.title("Zoning classification")
plt.show

# MSZoning by saleprice

# Violin plot showing house zone with Sale Price

sns.violinplot(train_df.MSZoning, train_df["SalePrice"])
plt.title("Saleprice by MSZoning")
plt.xlabel("MSZoning")
Plt.ylabel("Sale Price")

# Sale price per square foot

# Histogram showing Sale price per square foot with number sold

SalePrice=train_df['SalePrice']/train_df['GrLivArea']
plt.hist(SalePrice, color="blue")
plt.title("Sale Price Per Square Foot")
plt.ylabel('Number of Sales')
plt.xlabel('Price Per Square Foot')

# Sale Price based on Age of house

# Scatter plot showing age of house and sale price

ConstructionAge=train_df['YrSold']-train_df['YearBuilt']
plt.scatter(ConstructionAge, SalePrice, color="red")
plt.ylabel('price per square foot(in dollars)')
plt.xlabel("construction Age of house")

# Heating and AC relationship

# Stripp plot showing heating quality and sale price

sns.stripplot(x="HeatingQC", y="SalePrice", data=train_df, hue='HeatingQC', jitter=True, split=True)
plt.title("Sale Price based on Heating Quality")

# Above grade full bathrooms

# Boxplot showing Bathroom in relation to sale price

sns.boxplot(train_df["FullBath"], train_df["SalePrice"])
plt.title("Sale Price in relation to full Bathroom")

# Quality of Kitchen

# Factorplot showing Kitchen quality in relation to sale price

sns.factorplot("KitchenAbvGr", "SalePrice",  data=train_df, hue="KitchenQual")
plt.title("Sale Price in relation to Kitchen quality")


# # Correlation plot

import matplotlib.pyplot as plt

# Heatmap showing the relationship among variables

# Heatmap

import pandas as pd
import matplotlib.pyplot as plt
corr=train_num.corr()
plt.figure(figsize=(30,30))
sns.heatmap(corr[(corr>=0.5)|(corr<=-0.5)],
           cmap="YlGnBu", vmax=1.0, vmin=1.0,
           annot=True, annot_kws={"size": 15}, square=True)
plt.title('correlation between features')

# Data preparation for model 

# Data preparation for model

import numpy as np
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(final_data, target_log, test_size=0.30, random_state=0)
print("x_train", x_train.shape)
print("x_test", x_test.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)

# Build a base linear regression model

# Linear regression model

import statsmodels.api as sm

model1=sm.OLS(y_train, x_train).fit()

model1.summary()

# RMSE calculation

def rmse(predictions, targets):
    differences=predictions-targets
    differences_squared=differences ** 2
    mean_of_differences_squared=differences_squared.mean()
    rmse_val=np.sqrt(mean_of_differences_squared)
    return rmse_val

cols=['Model', 'R-Squared Value', 'Adj.R-Squared value', 'RMSE']

models_report=pd.DataFrame(columns=cols)

# Predicting the model on the test data

predictions1=model1.predict(x_test)

tmp1 = pd.Series({'Model': "Base Linear Regression Model",
               'R-Squared Value': model1.rsquared,
               'Adj.R-Squared Value':model1.rsquared_adj,
               'RMSE':rmse(predictions1, y_test)})
model1_report=models_report.append(tmp1,ignore_index=True)
model1_report

df_constant=sm.add_constant(final_data)
x_train1,x_test1,y_train1,y_test1=train_test_split(df_constant, target_log, test_size=0.30, random_state=0)

# Build linear regression model using statsmodel

import statsmodels.api as sm

model2=sm.OLS(y_train1, x_train1).fit()

model2.summary()

predictions2=model2.predict(x_test1)


tmp2 = pd.Series({'Model': " Linear Regression Model with constant",
               'R-Squared Value': model2.rsquared,
               'Adj.R-Squared Value':model2.rsquared_adj,
               'RMSE':rmse(predictions2, y_test)})
model2_report=models_report.append(tmp2,ignore_index=True)
model2_report

# VIF Calculation

print("\nvariance Inflation Factor")
cnames=x_train1.columns
for i in np.arange(0, len(cnames)):
    xvars=list(cnames)
    yvar=xvars.pop(i)
    mod=sm.OLS(x_train1[yvar],(x_train1[xvars]))
    res=mod.fit()
    vif=1/(1-res.rsquared)
    

print(yvar, round(vif,3))

# OLS regression

from sklearn.linear_model import LinearRegression

ols = LinearRegression()
ols.fit(x_train, y_train)
ols_yhat = ols.predict(x_test)

ols_yhat

# ridge regression

# Regression by Ridge method

ridge = Ridge(alpha = 0.5)
ridge.fit(x_train, y_train)
ridge_yhat = ridge.predict(x_test)

ridge_yhat

# Import libraries

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train,y_train)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from numpy import mean
from numpy import absolute

# kfolds cross validation

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

def rmsle(y_train, y_pred):
    return np.sqrt(mean_squared_error(y_train, y_pred))

def cv_rmse(model, x_train):
    rmse = np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)

# Setting alpha values

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.ensemble import StackingRegressor
import mlxtend
from mlxtend.regressor import StackingCVRegressor

import lightgbm 





import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor, LGBMClassifier, Booster

# Ridge regression

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

# Lasso regression

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))

# Elasticnet regression

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))   

# Support Vector regression

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))

# Gradient boosting regression

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)  

# lightgbm regression

lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=5000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )


import xgboost
from xgboost import XGBRegressor

# xgboost regression

xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)


# Stacking all models


stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)

# Printing rmse of the models

score = cv_rmse(ridge,x_train)
print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )


score = cv_rmse(ridge,x_train)
score = cv_rmse(lasso, x_train)
print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )


score = cv_rmse(elasticnet,x_train)
print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )


score = cv_rmse(svr,x_train)
print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )


score = cv_rmse(lightgbm,x_train)
print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )

score = cv_rmse(gbr,x_train)
print("gbr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )

score = cv_rmse(xgboost,x_train)
print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )






