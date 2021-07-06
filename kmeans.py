import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pycaret
from pycaret.clustering import *
from sklearn.cluster import KMeans

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
#KMEANS USING SKLEARN
def sklearn_elbow():
     wcss = []
     for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

def sklearn_Kmeans():
    kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(X)
    

##Pycaret method

def pycaret_kmeans(clusters_no):
    clust = setup(data = dataset)
    model=create_model('kmeans',num_clusters=clusters_no)
    res=assign_model(model)
    #print(res)
    plot_model(model)
    plot_model(model,plot='elbow')
    plot_model(model,plot='silhouette')


   
