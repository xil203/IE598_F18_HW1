
# coding: utf-8

# In[1]:


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
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df.columns = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash ','Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 
              'Proanthocyanins','Color intensity', 'PTRATIO', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
df.head()


# In[4]:


from sklearn.cross_validation import train_test_split 
from sklearn.preprocessing import StandardScaler
X = df.iloc[:, 1:].values 
y = df['Alcohol'].values 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)


# In[8]:


forest = RandomForestClassifier(criterion = 'gini',n_estimators = 25, random_state =1, n_jobs =2)
forest.fit(X_train_std, y_train)
forest.predict(X_test_std)

print('in sample RS CV score',cross_val_score(forest, X_train_std, y_train, cv=10))
#print('out of sample RS CV score',cross_val_score(forest, X_test_std, y_test, cv=10))
print('in sample accuracy score',forest.score(X_train_std, y_train))
#print('out of sample accuracy score',forest.score(X_test_std, y_test))


# In[9]:


forest = RandomForestClassifier(criterion = 'gini',n_estimators = 5, random_state =1, n_jobs =2)
forest.fit(X_train_std, y_train)
forest.predict(X_test_std)

print('in sample RS CV score',cross_val_score(forest, X_train_std, y_train, cv=10))
#print('out of sample RS CV score',cross_val_score(forest, X_test_std, y_test, cv=10))
print('in sample accuracy score',forest.score(X_train_std, y_train))
#print('out of sample accuracy score',forest.score(X_test_std, y_test))


# In[10]:


forest = RandomForestClassifier(criterion = 'gini',n_estimators = 10, random_state =1, n_jobs =2)
forest.fit(X_train_std, y_train)
forest.predict(X_test_std)

print('in sample RS CV score',cross_val_score(forest, X_train_std, y_train, cv=10))
#print('out of sample RS CV score',cross_val_score(forest, X_test_std, y_test, cv=10))
print('in sample accuracy score',forest.score(X_train_std, y_train))
#print('out of sample accuracy score',forest.score(X_test_std, y_test))


# In[11]:


forest = RandomForestClassifier(criterion = 'gini',n_estimators = 15, random_state =1, n_jobs =2)
forest.fit(X_train_std, y_train)
forest.predict(X_test_std)

print('in sample RS CV score',cross_val_score(forest, X_train_std, y_train, cv=10))
#print('out of sample RS CV score',cross_val_score(forest, X_test_std, y_test, cv=10))
print('in sample accuracy score',forest.score(X_train_std, y_train))
#print('out of sample accuracy score',forest.score(X_test_std, y_test))


# In[17]:


forest = RandomForestClassifier(criterion = 'gini',n_estimators = 20, random_state =1, n_jobs =2)
forest.fit(X_train_std, y_train)
forest.predict(X_test_std)

print('in sample RS CV score',cross_val_score(forest, X_train_std, y_train, cv=10))
#print('out of sample RS CV score',cross_val_score(forest, X_test_std, y_test, cv=10))
print('in sample accuracy score',forest.score(X_train_std, y_train))
#print('out of sample accuracy score',forest.score(X_test_std, y_test))


# In[18]:


from sklearn.ensemble import RandomForestClassifier
feat_labels = df.columns[1:]
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))


# In[19]:


plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),importances[indices],color='lightblue',align='center')
plt.xticks(range(X_train.shape[1]),feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()


# In[20]:


print("My name is Xiaodong Liu")
print("My NetID is: xl54")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

