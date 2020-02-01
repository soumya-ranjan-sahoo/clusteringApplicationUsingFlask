import pandas as pd
import os
from sklearn.cluster import AgglomerativeClustering, KMeans
import skfuzzy as fuzz
import numpy as np
from flask import Flask, request, jsonify, render_template,url_for
#from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from prettytable import PrettyTable
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from KMeds import *


#input_path     = 'Database-Objects.xlsx'
#clust_method   = 'fcm'
#apply_PCA      = True
#clust_num      = 5
home_path = os.getcwd() + '/'

class clustering(object):
    def __init__(self, clust_method, clust_num, apply_PCA, save_format = '.xlsx', affinity='euclidean', linkage='ward', random_state=0, fuzzy_param=2.0,error=1e-6,max_Iter=100, variance = 0.95, summary = True, download = True):
        self.clust_method = clust_method
        self.clust_num = clust_num
        self.affinity = affinity
        self.linkage = linkage
        self.download = download
        self.summary = summary
        self.save_format = save_format
        self.random_state = random_state
        self.apply_PCA = apply_PCA
        self.variance = variance
        self.fuzzy_param = fuzzy_param
        self.error = error
        self.max_Iter = max_Iter
        

    def read_file(self, path):
        filename, file_extension = os.path.splitext(path)
        if file_extension == '.xlsx':
            df = pd.read_excel(path, index_col=None, encoding = 'utf-8')
        if file_extension == '.csv':
            df = pd.read_csv(path, encoding = 'utf-8')
        return df

    def clean_data(self, dataframe):
        self.nm_df = dataframe.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        self.nm_df = self.nm_df.reset_index(drop=True)
        filt_df = self.nm_df._get_numeric_data()
        drop_column = filt_df.columns
        drop_column = drop_column[:5]
        filt_df  = filt_df.drop(columns = drop_column)
        print ("Cleaning")
        return filt_df


    def standardise_data(self, dataframe):
        stand_data = StandardScaler().fit_transform(dataframe)
        print("Data Standardization")
        return stand_data

    def agglomerative(self, filtered_df):
        print("agglomerative starts")
        cluster = AgglomerativeClustering(n_clusters=int(self.clust_num), affinity=self.affinity, linkage=self.linkage)
        res_clusters = cluster.fit_predict(filtered_df)
        print("agglomerative ends")
        return res_clusters

    def GaussianMixture(self, filtered_df):
        print("GMM starts")
        gmm = GaussianMixture(int(self.clust_num))
        res_cluster = gmm.fit_predict(filtered_df)
        print("GMM ends")
        return res_cluster

    def kmeans(self, filtered_df):
        print("kmeans starts")
        cluster = KMeans(n_clusters=int(self.clust_num), random_state=self.random_state)#.fit(X)
        res_cluster = cluster.fit_predict(filtered_df)
        print("kmeans ends")
        return res_cluster

    def fuzzyCmeans(self,filtered_df):
        print("fuzzyCmeans starts")
        center, membership, u0, d, jm, p, fpc = fuzz.cluster.cmeans(filtered_df,c=int(self.clust_num), m=self.fuzzy_param, error=self.error, maxiter=self.max_Iter)
        res_cluster = np.argmax(membership, axis=0)
        print("fuzzyCmeans ends")
        return res_cluster

    def kmedoids(self, filtered_df):
        print("kmedoids starts")
        #cluster = KMedoids(n_clusters=int(self.clust_num), random_state=self.random_state)#.fit(X)
        #res_cluster = cluster.fit_predict(filtered_df)
        res_cluster =K_Medoids_Clustering(filtered_df,clusters=int(self.clust_num))
        print("kmedoids ends")
        return res_cluster

    def download_data(self, clustered_df, file_format):
        if file_format == '.xlsx':
            print("saving as .xlsx format")
            clustered_df.to_excel(self.clust_method+'_output.xlsx')
        if file_format == '.csv':
            clustered_df.to_csv(self.clust_method+'_output.csv')

    def gen_summary(self,clustered_df):
        # if os.path.exists(self.clust_method+'_output.txt'):
            # os.remove(self.clust_method+'_output.txt')
        # self.summary_filename = self.clust_method+'_output.txt'
        # f = open(self.clust_method+'_output.txt', 'a')
        
        if os.path.exists(home_path+'summary/'+self.clust_method+self.clust_num+'.txt'):
            os.remove(home_path+'summary/'+self.clust_method+self.clust_num+'.txt')
        #self.summary_filename = self.clust_method+'_output.txt'
        f = open(home_path+'summary/'+self.clust_method+self.clust_num+'.txt','a')
        
        #print('The file is saved in {} format'.format(self.save_format), file=f)
        print('The clustering algorithm used is', self.clust_method, file=f)
        
        print('The input file have {} observations and {} features'.format(self.input_df.shape[0], self.input_df.shape[1]),file=f)
        print('Removed {} observations and {} features after preprocessing'.format(self.input_df.shape[0]-self.nm_df.shape[0], self.input_df.shape[1]-self.nm_df.shape[1]), file=f)
        if self.apply_PCA:
            print("Dimensionality reduction is enabled : {} features reduced to {} features".format(self.input_df.shape[1],self.n_components),file=f)
            print("%d components capture %f amount of variance in the data." %(self.n_components,self.variance), file=f)
        t = PrettyTable()
        t.field_names = ['Cluster#', 'Number of observations']
        for i in range(int(self.clust_num)):
            t.add_row([i, (clustered_df.output_class == i).sum()])
            #t.add_row([i, (self.clustered_df.output_class == i).sum()])
        print(t)
        f.write(str(t))
        f.close()

    def app_PCA(self, filt_df):
        print("PCA dimensionality reduction technique is applied")
        pca = PCA(self.variance)
        principalComponents = pca.fit_transform(filt_df)
        reduced_df = pd.DataFrame(principalComponents)
        self.n_components = pca.n_components_
        print("Projected number of components after PCA: %d"%self.n_components)
        return reduced_df

    def run_clustering(self, input_path):
        self.input_df = self.read_file(input_path)
        filt_df = self.clean_data(self.input_df)
        filt_df = self.standardise_data(filt_df)
        if self.apply_PCA:
            filt_df = self.app_PCA(filt_df)
        if self.clust_method == 'hierarchical clustering':
            clusters = self.agglomerative(filt_df)
        elif self.clust_method == 'k_means':
            clusters = self.kmeans(filt_df)
        elif self.clust_method == 'gmm_clustering':
            clusters = self.GaussianMixture(filt_df)
        elif self.clust_method == 'fuzzy_clustering':
            clusters = self.fuzzyCmeans(filt_df.T)
        elif self.clust_method == 'k_medoids':
            clusters = self.kmedoids(filt_df)
        clustered_df = pd.DataFrame(clusters, columns=['output_class'])
        resulted_df = pd.concat([self.nm_df[self.nm_df.columns[0:2]], clustered_df], axis = 1, sort=True)
        #if self.download:
            #print("saving the output file...")
            #self.download_data(clustered_df, clust_num, self.save_format)
        if self.summary:
            print("printing output summary...")
            self.gen_summary(clustered_df)
        return resulted_df

#path = 'Database_Objects.xlsx'
path = 'uploadedFile.csv'
def decision(clust_method,clust_num,apply_PCA):
#return render_template('cluster.html',k_text='You selected clusters')
    print("Starting clustering")
    cluster = clustering(clust_method = clust_method , clust_num = clust_num, apply_PCA = apply_PCA,download = True,summary = True)
    resulted_df = cluster.run_clustering(path)
    #resulted_df['output_class']=resulted_df.output_class.apply(lambda x: x+1)
    #download_data(output_clusters,'xlsx')
    return cluster,resulted_df

def download_data(cluster_obj,clustered_df, clust_num, file_format):
    print(clustered_df.head())
    print(cluster_obj.clust_method) # test
    if file_format == '.xlsx':
        clustered_df.to_excel(home_path+"output/output_"+cluster_obj.clust_method+"_"+ clust_num+ ".xlsx")
    if file_format == '.csv':
        clustered_df.to_csv(home_path+"output/output_"+cluster_obj.clust_method+"_"+ clust_num+ ".csv")
