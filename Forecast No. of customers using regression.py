# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 00:34:42 2021

@author: Sara Morani
"""

#%%1. IMPORT PACKAGES
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
%matplotlib inline

#%%2. GET THE DATA

dlent = pd.read_csv("data/lentil-data.csv")
dlent.info()
dlent.shape
dlent.describe()
# dlent.set_index('Restaurant')

dlent.isnull().any()

#%%3. READY THE DATA

#%%#3.1 DEFINE Y (RESPONSE) VARIABLE
y = dlent['Customers'] 


#%%#3.2 PREPARE CATEGORICAL VARIABLES
#DROP UNNEEDED COLUMNS

dX = dlent.drop(['Customers', 'Restaurant'], axis=1)
dX = pd.get_dummies(dX, drop_first=True)

# dumTheme=pd.get_dummies(dlent['Theme'],prefix='Theme')
# dX = pd.concat([dX,dumTheme],axis=1)
# dX.drop('Theme',axis=1, inplace=True)

dX2 = dX.drop(['Population', 'Theme_Veg', 'Student_Ratio', 'S_Asian_Ratio'],axis=1 )


#%%#3.3 DEFINE X (EXPLANATORY) VARIABLES

X=dX2


#%%Heatmap to see the correlation between independent variables

CorrMatrix = dX.corr()
print(CorrMatrix)

plt.figure(figsize =(10,10))
sns.heatmap(CorrMatrix, annot=True, cmap="coolwarm") #annot true shows value in heatmap
plt.savefig('fig\heatmap1.png', dpi=300)

#The plot shows there is no strong correlation between the independent variables. Also tried the below function to test the threshold of 80% for correlation between the independent variables. 


#with the following function we can select highly correlated features
#it will remove the first feature that is correlated with any other feature

def correlation(dataset, threshold):
    col_corr = set() #set of all names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])> threshold: #we are interested in absolute coeff value
                colname = corr_matrix.columns[i] #getting the name of column
                col_corr.add(colname)
    return col_corr           

corr_features = correlation(X, 0.8)
len(set(corr_features))

corr_features

#%%5. FIT A MODEL
#%%#5.1 USING STATSMODELS

X_sm = sm.add_constant(X)
smResults = sm.OLS(y, X_sm).fit()

#%%#5.2 USING SCI-KIT LEARN - just to check if results match

LinReg = LinearRegression()
LinReg.fit(X_sm,y)

LinReg.coef_
#array([   0.  , -250.19390794, 29.779478  , 648.45050221, 753.81970829])
LinReg.intercept_
#5988.577022894278

#getting  the same results as statsmodel


#%%6. EVALUATE

print(smResults.summary())

write_path = 'analysis\smResults.csv'
with open(write_path,'w') as f:
    f.write(smResults.summary().as_csv())

LRresult = (smResults.summary2().tables[1])

smResults.bse 
print(smResults.pvalues)
print(smResults.aic)
print(smResults.rsquared)
print(smResults.rsquared_adj)
smResults.params #returns coeff

smResults.scale
np.sqrt(smResults.scale)
#456.5798986182387


smfeat = ['const'] + X.columns.tolist()

coeff_dlent = pd.DataFrame(smResults.params, smfeat, columns=['Coefficient'])


