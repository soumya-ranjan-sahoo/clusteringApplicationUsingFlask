import os
import time
import pandas as pd
import numpy as np
import random
import operator
import math
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

np.random.seed(42)

class clustering(object):
    
    def __init__(self, method, nclusters ,fuzzy_param, max_Iter, error):
        self.method = method
        self.nclusters = nclusters
        self.fuzzy_param = fuzzy_param
        self.max_Iter = max_Iter
        self.error = error
        
        
    # Uploading the dataset    
    def upload_file(self, path):
        filename, file_extension = os.path.splitext(path)
        if file_extension == '.xlsx':
            dataFrame = pd.read_excel(path, index_col=None)
        if file_extension == '.csv':
            dataFrame = pd.read_csv(path) 
        
        
        return dataFrame
    
    
    #To clean the data: to check for missing values and remove unnecessary features
    def pre_Processing(self, dataFrame):
        self.data = dataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        self.data = self.data.reset_index(drop=True)
        self.Selected=self.data[self.data.columns[0:2]]       
        self.data = self.data.drop(self.data.columns[[0,1,2,3,4]], axis=1) 
        self.columns = list(self.data.columns)
        self.features = self.columns[:len(self.columns)]
        cleaned_Data= self.data[self.features]       
        return cleaned_Data
     
        
    #Initialize the membership probability
    def init_Membership_Prob(self,cleaned_Data,length):
        membership_matrix = list()
        for i in range(length):
            random_list = [random.random() for i in range(self.nclusters)]
            summ = sum(random_list)
            temp_list = [z/summ for z in random_list]
            membership_matrix.append(temp_list)
        return membership_matrix
    
    
    #To calculate centers of cluster
    def cluster_Center(self,membership_matrix,cleaned_Data,length):
        membership_val = list(zip(*membership_matrix))
        cluster_centers = list()
        for j in range(self.nclusters):
            x = list(membership_val[j])
            xraised = [e ** self.fuzzy_param for e in x]
            denominator = sum(xraised)
            temp_num = list()
            for i in range(length):
                data_point = list(cleaned_Data.iloc[i])
                prod = [xraised[i] * val for val in data_point]
                temp_num.append(prod)
            numerator = map(sum, zip(*temp_num))
            center = [z/denominator for z in numerator]
            cluster_centers.append(center)
        return cluster_centers
    
    
    
    #To update membership probabilitiy of every datapoint
    def update_Membership_Prob(self,membership_matrix,cluster_centers,length,cleaned_Data):
        p = float(2/(self.fuzzy_param-1))
        for i in range(length):
            x = list(cleaned_Data.iloc[i])
            distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(self.nclusters)]
            for j in range(self.nclusters):
                den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(self.nclusters)])
                membership_matrix[i][j] = float(1/den)       
        return membership_matrix
    
    
    
    #To get the maximum membership probability for every datapoint
    def get_Clusters(self,membership_matrix,length):
        cluster_labels = list()
        for i in range(length):
            max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_matrix[i]))
            cluster_labels.append(idx)
        return cluster_labels
    
    
    #Fuzzy clustering starts here
    def fuzzy_CMeans(self,cleaned_Data):
        # Membership Matrix
        length=len(cleaned_Data)
        niterations = 0

        membership_mat = self.init_Membership_Prob(cleaned_Data,length)
        mem_old = membership_mat
        
        while niterations < self.max_Iter - 1:
            mem_old = membership_mat.copy()
            mem_old1=np.array(mem_old)
            cluster_centers = self.cluster_Center(membership_mat,cleaned_Data,length)
            membership_mat = self.update_Membership_Prob(membership_mat, cluster_centers,length,cleaned_Data)
            membership_mat1=np.array(membership_mat)
            cluster_labels = self.get_Clusters(membership_mat,length)
            niterations += 1
            
            if np.linalg.norm(membership_mat1 - mem_old1) < self.error:
                break
        #Final error
        self.error = np.linalg.norm(membership_mat1 - mem_old1)
    
        return membership_mat,cluster_labels, cluster_centers
    
    
    
    #To save the result as xlsx or csv
    def save(self, clustered_df, file_format):
        if file_format == '.xlsx':
            clustered_df.to_excel('%s_output.xlsx'%self.method)
        if file_format == '.csv':
            clustered_df.to_csv('%s_output.csv'%self.method)      
       
    def pca(self, x):
    
        # Standardize the data to have a mean of 0 and a variance of 1
        X_std = StandardScaler().fit_transform(x)
        variance= 0.95
        pca = PCA(variance)  # returns features(components) which can cover variance of 0.95
        principalComponents = pca.fit_transform(X_std)
        reduced_data = pd.DataFrame(principalComponents)
        print("%d components capture %f amount of variance in the data." %(pca.n_components_,variance))
        return reduced_data
    
    
    def run_algorithm(self, path, download = True, pca = True, visualize = True, save_format = '.xlsx'):
        df = self.upload_file(path)
        cleaned_Data = self.pre_Processing(df)
        
        if pca:
            print('Dimensionality reduction applied')
            cleaned_Data = self.pca(cleaned_Data)
        membership_mat,cluster_labels, cluster_centers = self.fuzzy_CMeans(cleaned_Data)
        
        if download:
            clustering_Result = pd.DataFrame(membership_mat)                
            for i in range(self.nclusters):
                clustering_Result.rename(columns={ clustering_Result.columns[i]: "Cluster-"+ str(i+1) }, inplace = True)
            output_df = pd.concat([self.Selected, clustering_Result], axis = 1, sort=False)
            
            for i in range(len(cluster_labels)):
                cluster_labels[i] += 1
            output_df['Cluster(Max_membership)'] = cluster_labels
            self.save(output_df, save_format)
            
       
        return membership_mat,cluster_centers




path = 'Database-Objects.xlsx'    

cluster = clustering(method = 'fcm', nclusters = 3, fuzzy_param = 2.0, max_Iter = 20, error = 1e-5)


start=time.time()
membership_Prob,cluster_centers = cluster.run_algorithm(path, pca=True)
end=time.time()
print("Time = %d seconds"%(end-start))
