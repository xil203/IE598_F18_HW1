
# coding: utf-8

# In[72]:


from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


# In[79]:


iris = datasets.load_iris()
X = iris.data[:, :]#assign petal length and petal width(3rd and 4th columns) to fature matrix x
y = iris.target #assign corresponding class labels, 0,1 and 2 for three species, to vector y

X_train, X_test, y_train, y_test = train_test_split(
X,y,test_size = 0.1, random_state = 1, stratify = y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 5, random_state = 1)
tree.fit(X_train_std,y_train)

#print('in sample Logistic CV score',cross_val_score(lr, X_train_std, y_train, cv=3))
#print('out of sample Logistic CV score',cross_val_score(lr, X_test_std, y_test, cv=3))
print('in sample accuracy score',tree.score(X_train_std, y_train))
print('out of sample accuracy score',tree.score(X_test_std, y_test))


# In[80]:


tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 5, random_state = 2)
tree.fit(X_train_std,y_train)
print('in sample accuracy score',tree.score(X_train_std, y_train))
print('out of sample accuracy score',tree.score(X_test_std, y_test))


# In[81]:


RS_range = range(1,11)
tree_in_sample_scores = []
tree_out_sample_scores = []
for g in RS_range:
    tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 5, random_state = g)
    tree.fit(X_train_std,y_train)
    tree_in_sample_scores.append(tree.score(X_train_std, y_train))
    tree_out_sample_scores.append(tree.score(X_test_std, y_test))
    
print('tree_in_sample_scores from random state 1 to 10',tree_in_sample_scores)
print('tree_out_sample_scores from random state 1 to 10',tree_out_sample_scores)


# In[82]:


tree_mean_in_sample_score = np.mean(tree_in_sample_scores)
tree_mean_out_sample_score = np.mean(tree_out_sample_scores)
tree_mean_in_sample_score


# In[83]:


tree_mean_out_sample_score 


# In[84]:


np.std(tree_in_sample_scores)


# In[85]:


np.std(tree_out_sample_scores)


# In[86]:


print('in sample tree CV score',cross_val_score(tree, X_train_std, y_train, cv=10))
In_CV = cross_val_score(tree, X_train_std, y_train, cv=10)


# In[87]:


np.mean(In_CV)


# In[88]:


np.std(In_CV)


# In[52]:


X_test_std


# In[53]:


y_test


# In[54]:


# print('out sample tree CV score',cross_val_score(tree, X_test_std, y_test, cv=10))


# In[91]:


Out_CV = cross_val_score(tree, X_test_std, y_test, cv=5)
Out_CV 


# In[66]:


Out_CV = ([0.66666667, 1.        , 0.66666667, 1.        , 1.   , 0.66666667, 1.        , 0.66666667, 1.        , 1.])
Out_CV


# In[70]:


np.mean(Out_CV)


# In[71]:


np.std(Out_CV)


# In[75]:


Kfold = StratifiedKFold(n_splits = 10, random_state =1,).split(X_train_std, y_train)
scores = cross_val_score(estimator =tree, X=X_train_std, y=y_train, cv =10,n_jobs = 1)
print('CV accuracy scores: %s' %scores)


# In[77]:


Kfold = StratifiedKFold(n_splits = 10, random_state =1,).split(X_train_std, y_train)
out_scores = cross_val_score(estimator =tree, X= X_test_std, y= y_test, cv =10,n_jobs = 1)
print('CV accuracy scores: %s' %out_scores)


# In[92]:


print("My name is Xiaodong Liu")
print("My NetID is: xl54")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

