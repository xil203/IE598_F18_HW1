
# coding: utf-8

# In[140]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats
import pylab
import sys
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA


# In[30]:


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df.columns = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash ','Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 
              'Proanthocyanins','Color intensity', 'PTRATIO', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
df.head()


# In[25]:


df.describe()


# In[26]:


df.shape


# In[41]:


print('total number of wine1',len(df.loc[df['Alcohol'] == 1]))
print('total number of wine2',len(df.loc[df['Alcohol'] == 2]))
print('total number of wine3',len(df.loc[df['Alcohol'] == 3]))


# In[33]:


Ash = df.loc[:,'Ash']
stats.probplot(Ash, dist = "norm",plot = pylab)
pylab.title('Ash Probability Plot')
pylab.show()


# In[52]:


Mg1 = df.loc[:58,:]
wine1= df.loc[df['Alcohol'] == 1]
plt.scatter(Mg1,wine1)
plt.xlabel("Mg")
plt.ylabel("wine1")
plt.title("Attributes' Correlation plot for Mg vs wine1")
plt.show()


# In[57]:


Mg2 = df.loc[59:129,:]
wine2= df.loc[df['Alcohol'] == 2]
plt.scatter(Mg2,wine2)
plt.xlabel("Mg")
plt.ylabel("wine2")
plt.title("Attributes' Correlation plot for Mg vs wine2")
plt.show()


# In[60]:


Mg3 = df.loc[130:,:]
wine3= df.loc[df['Alcohol'] == 3]
plt.scatter(Mg3,wine3)
plt.xlabel("Mg")
plt.ylabel("wine3")
plt.title("Attributes' Correlation plot for Mg vs wine3")
plt.show()


# In[62]:


print('Check if there are missing values')
pd.isnull(df).sum()


# In[63]:


import numpy as np; np.random.seed(0)
import seaborn as sns
cols = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash ','Magnesium', 'Total phenols']
sns.pairplot(df[cols], size = 2.5)
plt.tight_layout()
plt.xlabel("attribute index")
plt.ylabel("attribute values")
plt.title("heatmap of cross-correlations of attributes")
plt.show()


# In[82]:


df1 = df.loc[:,'Malic acid']
plt.boxplot(df1,0,'gD')
plt.xlabel("attribute index")
plt.ylabel("values")
plt.title("boxplot of Malic acid")


# In[83]:


plt.boxplot(df,0,'gD')
plt.xlabel("attribute index")
plt.ylabel("values")
plt.title("boxplot of all attributes")


# In[91]:


from pandas.plotting import parallel_coordinates
plt.figure()
parallel_coordinates(df[cols], 'Alcohol')
plt.gca().legend_.remove()
plt.xlabel("attribute label")
plt.ylabel("values")
plt.title("parallel coordinate plot for Alcohol vs other features")


# In[93]:


import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.title('Heatmap of wine')
plt.show() 


# In[119]:


from sklearn.cross_validation import train_test_split 
from sklearn.preprocessing import StandardScaler
X = df.iloc[:, 1:].values 
y = df['Alcohol'].values 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)


# In[120]:



lr = LogisticRegression()
lr.fit(X_train_std, y_train)
lr.predict(X_test_std)
print('in sample Logistic CV score',cross_val_score(lr, X_train_std, y_train, cv=3))
print('out of sample Logistic CV score',cross_val_score(lr, X_test_std, y_test, cv=3))
print('in sample accuracy score',lr.score(X_train_std, y_train))
print('out of sample accuracy score',lr.score(X_test_std, y_test))


# In[121]:


svm = SVC(kernel = 'linear', C = 1.0, random_state=1)
svm.fit(X_train_std,y_train)
svm.predict(X_test_std)
print('in sample Logistic CV score',cross_val_score(svm, X_train_std, y_train, cv=3))
print('out of sample Logistic CV score',cross_val_score(svm, X_test_std, y_test, cv=3))
print('in sample accuracy score',svm.score(X_train_std, y_train))
print('out of sample accuracy score',svm.score(X_test_std, y_test))


# In[134]:


pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca,y_train)
lr.predict(X_test_pca)
print('in sample accuracy score',lr.score(X_train_pca, y_train))
print('out of sample accuracy score',lr.score(X_test_pca, y_test))


# In[137]:


svm.fit(X_train_pca,y_train)
svm.predict(X_test_pca)
print('in sample accuracy score',svm.score(X_train_pca, y_train))
print('out of sample accuracy score',svm.score(X_test_pca, y_test))


# In[136]:


lda = LDA(n_components = 2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
lr.fit(X_train_lda, y_train)
lr.predict(X_test_lda)
print('in sample accuracy score',lr.score(X_train_lda, y_train))
print('out of sample accuracy score',lr.score(X_test_lda, y_test))


# In[138]:


svm.fit(X_train_lda, y_train)
svm.predict(X_test_lda)
print('in sample accuracy score',svm.score(X_train_lda, y_train))
print('out of sample accuracy score',svm.score(X_test_lda, y_test))


# In[142]:


kpca = KernelPCA(n_components=2, kernel = 'rbf', gamma =15)
X_train_kpca = kpca.fit_transform(X_train_std, y_train)
X_test_kpca = kpca.transform(X_test_std)
lr.fit(X_train_kpca, y_train)
lr.predict(X_test_kpca)
print('in sample accuracy score',lr.score(X_train_kpca, y_train))
print('out of sample accuracy score',lr.score(X_test_kpca, y_test))


# In[143]:


svm.fit(X_train_kpca, y_train)
svm.predict(X_test_kpca)
print('in sample accuracy score',svm.score(X_train_kpca, y_train))
print('out of sample accuracy score',svm.score(X_test_kpca, y_test))


# In[151]:


gamma_range = range(1,20)
lr_in_sample_scores = []
lr_out_sample_scores = []
for g in gamma_range:
    kpca = KernelPCA(n_components=2, kernel = 'rbf', gamma =g)
    X_train_kpca = kpca.fit_transform(X_train_std, y_train)
    X_test_kpca = kpca.transform(X_test_std)
    lr.fit(X_train_kpca, y_train)
    lr_in_sample_scores.append(lr.score(X_train_kpca, y_train))
    lr_out_sample_scores.append(lr.score(X_test_kpca, y_test))
print('lr_in_sample_scores from gamma =1 to gamma =20',lr_in_sample_scores)
print('lr_out_sample_scores from gamma =1 to gamma =20',lr_out_sample_scores)


# In[152]:


gamma_range = range(1,20)
svm_in_sample_scores = []
svm_out_sample_scores = []
for g in gamma_range:
    kpca = KernelPCA(n_components=2, kernel = 'rbf', gamma =g)
    X_train_kpca = kpca.fit_transform(X_train_std, y_train)
    X_test_kpca = kpca.transform(X_test_std)
    svm.fit(X_train_kpca, y_train)
    svm_in_sample_scores.append(svm.score(X_train_kpca, y_train))
    svm_out_sample_scores.append(svm.score(X_test_kpca, y_test))
print('lr_in_sample_scores from gamma =1 to gamma =20',svm_in_sample_scores)
print('lr_out_sample_scores from gamma =1 to gamma =20',svm_out_sample_scores)


# In[153]:


print("My name is Xiaodong Liu")
print("My NetID is: xl54")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

