import os
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



np.random.seed(42)

class clustering(object):
    
    def __init__(self, method, n_clusters ,fuzzy_param, max_Iter, error,variance = 0.95):
        self.method = method
        self.n_clusters = n_clusters
        self.fuzzy_param = fuzzy_param
        self.max_Iter = max_Iter
        self.error = error
        self.variance=variance
        
        
    # Uploading the dataset    
    def upload_file(self,path):
        filename, file_extension = os.path.splitext(path)
        if file_extension == '.xlsx':
            dataFrame = pd.read_excel(path, index_col=None)
        if file_extension == '.csv':
            dataFrame = pd.read_csv(path)  
        else:
            dataFrame = pd.read_excel(path, index_col=None)
        
        return dataFrame
    
    
    #To clean the data: to check for missing values and remove unnecessary features
    def pre_Processing(self, dataframe):
        self.nm_df = dataframe.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        self.nm_df = self.nm_df.reset_index(drop=True)
        filt_df = self.nm_df._get_numeric_data()
        return filt_df

        
    #Initialize the membership probability
    def fuzzyCmeans(self,cleaned_Data):

        center, membership, u0, d, jm, p, fpc = fuzz.cluster.cmeans(cleaned_Data,c=self.n_clusters, m=self.fuzzy_param, error=self.error, maxiter=self.max_Iter)
        cluster_membership = np.argmax(membership, axis=0)

        return cluster_membership
    
    #To save the result as xlsx or csv
    def save(self, clustered_df, file_format):
        if file_format == '.xlsx':
            clustered_df.to_excel('%s_output.xlsx'%self.method)
        if file_format == '.csv':
            clustered_df.to_csv('%s_output.csv'%self.method) 
            
            
    def standardise_data(self, dataframe):
        stand_data = StandardScaler().fit_transform(dataframe)
        return stand_data


    def pca(self, filt_df):
        pca = PCA(self.variance)
        principalComponents = pca.fit_transform(filt_df)
        reduced_df = pd.DataFrame(principalComponents)
        self.n_components = pca.n_components_
        return reduced_df


    
    
    def run_algorithm(self, path, download = True,pca=False, save_format = '.xlsx'):
        df = self.upload_file(path)
        cleaned_Data = self.pre_Processing(df)
        cleaned_Data = self.standardise_data(cleaned_Data)      
               
        clusters = self.fuzzyCmeans(cleaned_Data.T) 
        clustered_df = pd.DataFrame(clusters, columns=['output_class'])
        self.clustered_df = pd.concat([self.nm_df[self.nm_df.columns[0:2]], clustered_df], axis = 1, sort=True)


        if download:
           self.save(self.clustered_df, save_format)
            
        return clusters


path = 'Database-Objects.xlsx'    

cluster = clustering(method = 'fcm', n_clusters = 3, fuzzy_param = 2.0, max_Iter=100, error = 1e-5)


output_clusters = cluster.run_algorithm(path)
