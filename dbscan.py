import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

X = pd.read_csv('CC GENERAL.csv')
X = X.drop('CUST_ID', axis = 1)
X.fillna(method ='ffill', inplace = True)

#DBSCAN USING SKLEARN

def sklearn_dbscan():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_normalized = normalize(X_scaled)
    X_normalized = pd.DataFrame(X_normalized)
    pca = PCA(n_components = 2)
    X_principal = pca.fit_transform(X_normalized)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']
    db = DBSCAN(eps = 0.0375, min_samples = 3).fit(X_principal)
    labels = db.labels_
 
#DBSCAN USING PYCARET

def pycaret_dbscan(eps_value,min_sample_size):
    clust = setup(data = dataset,normalize=True)
    model=create_model('dbscan',eps=eps_value,min_samples=min_sample_size)
    res=assign_model(model)
    plot_model(model)
