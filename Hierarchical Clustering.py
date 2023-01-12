import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# To perform Hierarchical clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# import all libraries and dependencies for machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan

mall = pd.read_csv("Supermarket_Customers_data_v2.csv", delimiter=';')
mall = mall.rename(columns={"Spending Score (1-100): Score assigned by the shop based on customer behavior and spending nature": "Spending Score (1-100)"})
print(mall.head())

print(mall.shape)
print(mall.info())
print(mall.describe())


# Gender graph
plt.figure(figsize = (5,5))
sex = mall['Sex'].sort_values(ascending = False)
ax = sns.countplot(x='Sex', data= mall)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.xticks(rotation=90)
plt.show()

# Age graph
plt.figure(figsize = (20,5))
gender = mall['Age'].sort_values(ascending = False)
ax = sns.countplot(x='Age', data= mall)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.show()

# Annual Income graph
plt.figure(figsize = (25,5))
gender = mall['Annual Income (k$)'].sort_values(ascending = False)
ax = sns.countplot(x='Annual Income (k$)', data= mall)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.show()

#Spending score graph
plt.figure(figsize = (27,5))
gender = mall['Spending Score (1-100)'].sort_values(ascending = False)
ax = sns.countplot(x='Spending Score (1-100)', data= mall)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.show()

# Let's check the correlation coefficients to see which variables are highly correlated
mall_without_id = mall.drop(['Customer_ID'], axis = 1)
plt.figure(figsize = (10,8))
sns.heatmap(mall_without_id.corr(), annot = True, cmap="rainbow", fmt=".2f", annot_kws={"size": 15})
plt.show()

# Pairplot graph
sns.pairplot(mall,corner=True,diag_kind="kde")
plt.show()

#Outlier Analysis
f, axes = plt.subplots(1,3, figsize=(15,5))
s=sns.violinplot(y=mall.Age,ax=axes[0])
axes[0].set_title('Age')
s=sns.violinplot(y=mall['Annual Income (k$)'],ax=axes[1])
axes[1].set_title('Annual Income (k$)')
s=sns.violinplot(y=mall['Spending Score (1-100)'],ax=axes[2])
axes[2].set_title('Spending Score (1-100)')
plt.show()

#We use Percentile Capping (Winsorization) for outliers handling¶
Q3 = mall['Annual Income (k$)'].quantile(0.99)
Q1 = mall['Annual Income (k$)'].quantile(0.01)
mall['Annual Income (k$)'][mall['Annual Income (k$)']<=Q1]=Q1
mall['Annual Income (k$)'][mall['Annual Income (k$)']>=Q3]=Q3
print(mall.describe())

# Outlier Analysis 2
f, axes = plt.subplots(1,3, figsize=(15,5))
s=sns.violinplot(y=mall.Age,ax=axes[0])
axes[0].set_title('Age')
s=sns.violinplot(y=mall['Annual Income (k$)'],ax=axes[1])
axes[1].set_title('Annual Income (k$)')
s=sns.violinplot(y=mall['Spending Score (1-100)'],ax=axes[2])
axes[2].set_title('Spending Score (1-100)')
plt.show()

# Dropping CustomerID field to form cluster
mall_c = mall.drop(['Customer_ID'],axis=1,inplace=True)
print(mall.head)


#Hopkins Statistic Test (Factorize Sex)
mall['Sex'] = pd.factorize(mall.Sex)[0]
print(mall.head())
def hopkins(X):
    d = X.shape[1]
    n = len(X)
    m = int(0.1 * n)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

    rand_X = sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0), np.amax(X, axis=0), d).reshape(1, -1), 2,
                                    return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    HS = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(HS):
        print(ujd, wjd)
        HS = 0

    return HS

# Hopkins score
Hopkins_score=round(hopkins(mall),2)
print("{} is a good Hopkins score for Clustering.".format(Hopkins_score))





# Single linkage Clustoring
plt.figure(figsize = (20,10))
mergings = linkage(mall, method='single',metric='euclidean')
dendrogram(mergings)
plt.show()

# Complete Linkage Clustoring
plt.figure(figsize = (5,10))
mergings = linkage(mall, method='complete',metric='euclidean')
dendrogram(mergings)
plt.show()


# 4 clusters
cluster_labels = cut_tree(mergings, n_clusters=4).reshape(-1, )
# Assign the label
mall['Cluster_Id'] = cluster_labels
# Number of customers in each cluster
mall['Cluster_Id'].value_counts(ascending=True)
print(mall.columns)

#Plot Cluster Graph
plt.figure(figsize = (20,6))
plt.subplot(1,3,1)
sns.scatterplot(x = 'Age', y = 'Annual Income (k$)',hue='Cluster_Id',data = mall,legend='full',palette="Set1")
plt.subplot(1,3,2)
sns.scatterplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)',hue='Cluster_Id', data = mall,legend='full',palette="Set1")
plt.subplot(1,3,3)
sns.scatterplot(x = 'Spending Score (1-100)', y = 'Age',hue='Cluster_Id',data= mall,legend='full',palette="Set1")
plt.show()

#Violin plot on Original attributes to visualize the spread of the data

fig, axes = plt.subplots(1,3, figsize=(20,5))

sns.violinplot(x = 'Cluster_Id', y = 'Age', data = mall,ax=axes[0])
sns.violinplot(x = 'Cluster_Id', y = 'Annual Income (k$)', data = mall,ax=axes[1])
sns.violinplot(x = 'Cluster_Id', y = 'Spending Score (1-100)', data=mall,ax=axes[2])
plt.show()

#Group of Clusters
mall[['Age', 'Annual Income (k$)','Spending Score (1-100)','Cluster_Id']].groupby('Cluster_Id').mean()

#Cluster 0
cluster_0= mall[mall['Cluster_Id']==0]
cluster_0['Sex'] = cluster_0['Sex'].replace({0: 'male', 1: 'female'})
fig, axes = plt.subplots(1,3, figsize=(20,5))

sns.violinplot(x = 'Sex', y = 'Age', data = cluster_0,ax=axes[0])
sns.violinplot(x = 'Sex', y = 'Annual Income (k$)', data = cluster_0,ax=axes[1])
sns.violinplot(x = 'Sex', y = 'Spending Score (1-100)', data=cluster_0,ax=axes[2])
plt.show()

#Cluster 1
cluster_1= mall[mall['Cluster_Id']==1]
cluster_1['Sex'] = cluster_1['Sex'].replace({0: 'male', 1: 'female'})
fig, axes = plt.subplots(1,3, figsize=(20,5))

sns.violinplot(x = 'Sex', y = 'Age', data = cluster_1,ax=axes[0])
sns.violinplot(x = 'Sex', y = 'Annual Income (k$)', data = cluster_1,ax=axes[1])
sns.violinplot(x = 'Sex', y = 'Spending Score (1-100)', data=cluster_1,ax=axes[2])
plt.show()

#Cluster 2
cluster_2= mall[mall['Cluster_Id']==2]
cluster_2['Sex'] = cluster_2['Sex'].replace({0: 'male', 1: 'female'})
fig, axes = plt.subplots(1,3, figsize=(20,5))

sns.violinplot(x = 'Sex', y = 'Age', data = cluster_2,ax=axes[0])
sns.violinplot(x = 'Sex', y = 'Annual Income (k$)', data = cluster_2,ax=axes[1])
sns.violinplot(x = 'Sex', y = 'Spending Score (1-100)', data=cluster_2,ax=axes[2])
plt.show()

#Cluster 3
cluster_3= mall[mall['Cluster_Id']==3]
cluster_3['Sex'] = cluster_3['Sex'].replace({0: 'male', 1: 'female'})
fig, axes = plt.subplots(1,3, figsize=(20,5))

sns.violinplot(x = 'Sex', y = 'Age', data = cluster_3,ax=axes[0])
sns.violinplot(x = 'Sex', y = 'Annual Income (k$)', data = cluster_3,ax=axes[1])
sns.violinplot(x = 'Sex', y = 'Spending Score (1-100)', data=cluster_3,ax=axes[2])
plt.show()