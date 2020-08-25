#[Anuj Ashok Kolekar]
#[20226315]
#[MMA]
#[2021W]
#[MMA 869]
#[16th August 2020]

#Answer to Question 1 ,Part 1[a]

#Importing the required libraries and dataset


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.datasets import load_nfl
from scipy.spatial import distance

data=pd.read_csv('C:/Users/anuj/Documents/Anuj/MMA/Maching Learning & AI/Individual Assignment/jewelry_customers.csv')

data.head()

data.tail()

data.describe()




# In[2]:


#Validating if there is any missing data 

pd.isna(data).sum()


# In[3]:


#Scaling the features as values for each feature are distributed across wide range

scl=StandardScaler()
data1=scl.fit_transform(data)


# In[4]:


pd.DataFrame(data1).describe()


# In[5]:

#Answer to Question 1 ,Part 1[b]

#Running the kmeans algorithm (k ranging from 2 to 9)
#Elbow method implementation 
#Generating Inertia and Silhouette score for each iteration 

cluster=[]
Inertia=[]
silscore=[]

x=data1

for k in range(2,9):
    kmeans=KMeans(n_clusters=k,n_init=25,random_state=40,max_iter=1000,init='k-means++')

    y_pred=kmeans.fit_predict(x)
    
    cluster.append(k)
    Inertia.append(kmeans.inertia_)
    
    silscore.append(silhouette_score(x,kmeans.labels_))

    
    print("\n"'Number of clusters:',k,"\n"'Clustering:',y_pred,"\n"'Inertia with',k,'Clusters:',kmeans.inertia_,"\n"
          'Silhouette_score with',k,'Clusters:',silhouette_score(x,kmeans.labels_),"\n"'Centroids:',kmeans.cluster_centers_) 
    
   



plt.plot(cluster,Inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Inertia Graph')
plt.show()

plt.plot(cluster,silscore)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhoutte Graph')
plt.show()





# In[18]:


#Elbow method indicates that inertia starts increasing rapidly after k=5
#Highest Silhouette score is observed for k =5 
#Generating the clusters for k =5
#Plotting the clusters using two features 'Income' and 'Spending Score'

kmeans=KMeans(n_clusters=5,n_init=25,random_state=40,init='k-means++')

y_pred=kmeans.fit(x)

plt.figure;

plt.scatter(x[:,1], x[:,2], s=100, c=y_pred.labels_)

plt.scatter(
y_pred.cluster_centers_[:, 1], y_pred.cluster_centers_[:, 2],
c='red', edgecolor='black',
label='centroids'
)

plt.xlabel('Income')
plt.ylabel('SpendingScore')


# In[19]:


#Silhouette Score Map for each element of every cluster

visualizer=SilhouetteVisualizer(kmeans)
visualizer.fit(x)
visualizer.poof()

fig=visualizer.ax.get_figure()


# In[20]:


#Publishing the cluster centers
kmeans.cluster_centers_


# In[21]:


#Publishing number of customers in each cluster

z_pred=kmeans.fit_predict(x)

data3=pd.DataFrame(data1)


data3['cluster']=z_pred

data3['cluster'].value_counts()


# In[22]:

#Answer to Question 1 ,Part 1[c]
#Summary cluster statistics 

for label in set(kmeans.labels_):
    print('\nCluster{}:'.format(label))
    
    print(data[kmeans.labels_==label].describe())
    
    
    


# In[23]:


#Generating the details of Cluster representative (Personna)


for i, label in enumerate(set(kmeans.labels_)):    
    data_tmp = data[kmeans.labels_==label].copy()
    
    exemplar_idx = distance.cdist([kmeans.cluster_centers_[i]], data_tmp).argmin()
    exemplar = pd.DataFrame(data_tmp.iloc[exemplar_idx])
   
    print('\nCluster {}:'.format(label))
    
    
    print(exemplar)
    
    

    

