# cluster_explanations

## Description :
Typically in clustering or unsupervised learning all observations are assigned with cluster labels/id and it is required to understand the cluster at a high level w.r.to the data attributes . In supervised learning  few existing packages already provide explanation/rules why a particular class label is assigned to data record. For un-supervised learning we can do similar explainability analysis using state of the art work.

## Requirements :
python , data structures, collections, concurrency , Numpy, clustering algorithms , information theory metrics 

## Outcomes :
Open source contribution to one of (sci-kit learn , pyspark , pycaret) , short paper/blog

## Observations:
* Applying supervised learning on cluster groups with assigned cluster labels can also yields interpretable results but supervised learning is error-prone. 


## To-dos:
* Identify datasets(3) with some variety (categorical and continuous data) (S)
* Code for clustering using DBSCAN and K_means algorithm using sklearn, pycaret (S)
* Code for Entropy and Gini partition rules for cluster groups 
* Code for Association rule mining to compare cluster group rules,comparison of cluster group centriods
* Explanation rule about cluster groups that uniquely identifies each cluster