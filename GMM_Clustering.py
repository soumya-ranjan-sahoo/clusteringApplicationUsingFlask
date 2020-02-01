import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#from pandas import read_excel
#from pandas import DataFrame 
from sklearn.mixture import GaussianMixture

class GMM_Clustering(object):
    def Gmm(self,filt_df, clust_num):
        X=filt_df
        n_components =clust_num   # enter the number of clusters required
        gmm = GaussianMixture(n_components)
        # Fit the GMM model for the dataset  
        # which expresses the dataset as a  
        # mixture of Gaussian Distribution 
        gmm.fit(X)
        # Assign a label to each sample 
        labels = gmm.predict(X)
        X['labels']= labels
        return X
        #for i in range(n_components):
        #    with open('Dataframe.csv', 'a', newline ="") as f:
        #        Y = X[X['labels']== i]
        #        f.write(Y.to_csv(header=None))
        #        f.close()