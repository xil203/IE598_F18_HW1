
# coding: utf-8

# In[126]:


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
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch10/housing.data.txt',
                 header=None, 
                 sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
#print(df.shape)
df.head()


# In[19]:


df.describe()


# In[3]:


df.shape


# In[16]:


CRIM = df.loc[:,'CRIM']
stats.probplot(CRIM, dist = "norm",plot = pylab)
pylab.show()


# In[27]:


scaler = StandardScaler().fit(df)
df1 = scaler.transform(df)
ZN = df.loc[:,'ZN']
MEDV= df.loc[:,'MEDV']
plt.scatter(ZN,MEDV)
plt.xlabel("ZN")
plt.ylabel("MEDV")
plt.title("Attributes' Correlation plot for Zn vs MedV")
plt.show()


# In[20]:


pd.isnull(df).sum()


# In[85]:


import numpy as np; np.random.seed(0)
import seaborn as sns
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size = 2.5)
plt.tight_layout()
plt.xlabel("attribute index")
plt.ylabel("attribute values")
plt.title("heatmap of cross-correlations of attributes")
plt.show()


# In[50]:


plt.boxplot(df1,0,'gD')
plt.xlabel("attribute index")
plt.ylabel("values")
plt.title("boxplot of attributes")


# In[70]:


from pandas.plotting import parallel_coordinates
plt.figure()
parallel_coordinates(df, 'MEDV')
plt.gca().legend_.remove()
plt.xlabel("attribute label")
plt.ylabel("values")
plt.title("parallel coordinate plot for attributes")


# In[86]:


import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show() 


# In[134]:


class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []
        
        for i in range(self.n_iter):            
            output = self.net_input(X)            
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self
    
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):        
        return self.net_input(X)
    
X = df[['RM']].values
y = df['MEDV'].values 
sc_x = StandardScaler() 
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()


def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='blue') 
    plt.plot(X, model.predict(X), color='red')  
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[139]:


class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []
        
        for i in range(self.n_iter):            
            output = self.net_input(X)            
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self
    
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):        
        return self.net_input(X)
    
X = df[['CRIM']].values
y = df['MEDV'].values 
sc_x = StandardScaler() 
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()


def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='blue') 
    plt.plot(X, model.predict(X), color='red')  
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[143]:


class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []
        
        for i in range(self.n_iter):            
            output = self.net_input(X)            
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self
    
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):        
        return self.net_input(X)
    
X = df[['ZN']].values
y = df['MEDV'].values 
sc_x = StandardScaler() 
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()


def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='blue') 
    plt.plot(X, model.predict(X), color='red')  
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [ZN] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[144]:


class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []
        
        for i in range(self.n_iter):            
            output = self.net_input(X)            
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self
    
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):        
        return self.net_input(X)
    
X = df[['INDUS']].values
y = df['MEDV'].values 
sc_x = StandardScaler() 
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()


def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='blue') 
    plt.plot(X, model.predict(X), color='red')  
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [INDUS] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[145]:


class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []
        
        for i in range(self.n_iter):            
            output = self.net_input(X)            
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self
    
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):        
        return self.net_input(X)
    
X = df[['CHAS']].values
y = df['MEDV'].values 
sc_x = StandardScaler() 
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()


def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='blue') 
    plt.plot(X, model.predict(X), color='red')  
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [CHAS] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[150]:


class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []
        
        for i in range(self.n_iter):            
            output = self.net_input(X)            
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self
    
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):        
        return self.net_input(X)
    
X = df[['NOX']].values
y = df['MEDV'].values 
sc_x = StandardScaler() 
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()


def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='blue') 
    plt.plot(X, model.predict(X), color='red')  
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [NOX] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[151]:


class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []
        
        for i in range(self.n_iter):            
            output = self.net_input(X)            
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self
    
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):        
        return self.net_input(X)
    
X = df[['RM']].values
y = df['MEDV'].values 
sc_x = StandardScaler() 
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()


def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='blue') 
    plt.plot(X, model.predict(X), color='red')  
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[152]:


class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []
        
        for i in range(self.n_iter):            
            output = self.net_input(X)            
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self
    
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):        
        return self.net_input(X)
    
X = df[['RM']].values
y = df['MEDV'].values 
sc_x = StandardScaler() 
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()


def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='blue') 
    plt.plot(X, model.predict(X), color='red')  
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[153]:


class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []
        
        for i in range(self.n_iter):            
            output = self.net_input(X)            
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self
    
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):        
        return self.net_input(X)
    
X = df[['AGE']].values
y = df['MEDV'].values 
sc_x = StandardScaler() 
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()


def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='blue') 
    plt.plot(X, model.predict(X), color='red')  
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [AGE] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[154]:


class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []
        
        for i in range(self.n_iter):            
            output = self.net_input(X)            
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self
    
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):        
        return self.net_input(X)
    
X = df[['DIS']].values
y = df['MEDV'].values 
sc_x = StandardScaler() 
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()


def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='blue') 
    plt.plot(X, model.predict(X), color='red')  
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [DIS] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[155]:


class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []
        
        for i in range(self.n_iter):            
            output = self.net_input(X)            
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self
    
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):        
        return self.net_input(X)
    
X = df[['RAD']].values
y = df['MEDV'].values 
sc_x = StandardScaler() 
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()


def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='blue') 
    plt.plot(X, model.predict(X), color='red')  
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RAD] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[156]:


class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []
        
        for i in range(self.n_iter):            
            output = self.net_input(X)            
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self
    
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):        
        return self.net_input(X)
    
X = df[['TAX']].values
y = df['MEDV'].values 
sc_x = StandardScaler() 
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()


def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='blue') 
    plt.plot(X, model.predict(X), color='red')  
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [TAX] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[157]:


class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []
        
        for i in range(self.n_iter):            
            output = self.net_input(X)            
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self
    
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):        
        return self.net_input(X)
    
X = df[['PTRATIO']].values
y = df['MEDV'].values 
sc_x = StandardScaler() 
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()


def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='blue') 
    plt.plot(X, model.predict(X), color='red')  
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [PTRATIO] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[158]:


class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []
        
        for i in range(self.n_iter):            
            output = self.net_input(X)            
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self
    
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):        
        return self.net_input(X)
    
X = df[['B']].values
y = df['MEDV'].values 
sc_x = StandardScaler() 
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()


def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='blue') 
    plt.plot(X, model.predict(X), color='red')  
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [B] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[159]:


class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):        
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):        
        self.w_ = np.zeros(1 + X.shape[1])        
        self.cost_ = []
        
        for i in range(self.n_iter):            
            output = self.net_input(X)            
            errors = (y - output)            
            self.w_[1:] += self.eta * X.T.dot(errors)            
            self.w_[0] += self.eta * errors.sum()            
            cost = (errors**2).sum() / 2.0            
            self.cost_.append(cost)        
        return self
    
    def net_input(self, X):        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):        
        return self.net_input(X)
    
X = df[['LSTAT']].values
y = df['MEDV'].values 
sc_x = StandardScaler() 
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X) 
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD() 
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_) 
plt.ylabel('SSE') 
plt.xlabel('Epoch') 
plt.show()


def lin_regplot(X, y, model): 
    plt.scatter(X, y, c='blue') 
    plt.plot(X, model.predict(X), color='red')  
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [LSTAT] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[146]:


from sklearn.cross_validation import train_test_split 
X = df.iloc[:, :-1].values 
y = df['MEDV'].values 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0) 
slr = LinearRegression() 
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train,c='blue', marker='o', label='Training data') 
plt.scatter(y_test_pred,  y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values') 
plt.ylabel('Residuals')
plt.title('Linear Regression')
plt.legend(loc='upper left') 
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red') 
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error 
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))


# In[188]:



from sklearn.cross_validation import train_test_split 
X = df.iloc[:, :-1].values 
y = df['MEDV'].values 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0) 
lasso = Lasso(alpha =0.00000000000000001) 
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train,c='blue', marker='o', label='Training data') 
plt.scatter(y_test_pred,  y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values') 
plt.ylabel('Residuals') 
plt.title('Lasso Regression')
plt.legend(loc='upper left') 
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red') 
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error 
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))


# In[187]:



from sklearn.cross_validation import train_test_split 
X = df.iloc[:, :-1].values 
y = df['MEDV'].values 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0) 
lasso = Lasso(alpha =1) 
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train,c='blue', marker='o', label='Training data') 
plt.scatter(y_test_pred,  y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values') 
plt.ylabel('Residuals') 
plt.title('Lasso Regression')
plt.legend(loc='upper left') 
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red') 
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error 
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))


# In[185]:


from sklearn.cross_validation import train_test_split 
X = df.iloc[:, :-1].values 
y = df['MEDV'].values 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0) 
ridge = Ridge(alpha = 1) 
ridge.fit(X_train, y_train)
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train,c='blue', marker='o', label='Training data') 
plt.scatter(y_test_pred,  y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values') 
plt.ylabel('Residuals') 
plt.title('Ridge')
plt.legend(loc='upper left') 
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red') 
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error 
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))


# In[186]:


from sklearn.cross_validation import train_test_split 
X = df.iloc[:, :-1].values 
y = df['MEDV'].values 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0) 
ridge = Ridge(alpha = 2) 
ridge.fit(X_train, y_train)
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train,c='blue', marker='o', label='Training data') 
plt.scatter(y_test_pred,  y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values') 
plt.ylabel('Residuals') 
plt.title('Ridge')
plt.legend(loc='upper left') 
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red') 
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error 
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))


# In[171]:


X = df.iloc[:, :-1].values 
y = df['MEDV'].values 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0) 
enet = ElasticNet(alpha = 0.10,l1_ratio = 1) 
enet.fit(X_train, y_train)
y_train_pred = enet.predict(X_train)
y_test_pred = enet.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train,c='blue', marker='o', label='Training data') 
plt.scatter(y_test_pred,  y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values') 
plt.ylabel('Residuals') 
plt.title('ElasticNet')
plt.legend(loc='upper left') 
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red') 
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error 
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))


# In[183]:


X = df.iloc[:, :-1].values 
y = df['MEDV'].values 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0) 
enet = ElasticNet(alpha = 0.10,l1_ratio = 0) 
enet.fit(X_train, y_train)
y_train_pred = enet.predict(X_train)
y_test_pred = enet.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train,c='blue', marker='o', label='Training data') 
plt.scatter(y_test_pred,  y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values') 
plt.ylabel('Residuals') 
plt.title('ElasticNet')
plt.legend(loc='upper left') 
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red') 
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error 
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))


# In[193]:


print('My name is Xiaodong Liu')
print('My NetID is xl54')
print('I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation')

