import numpy as np
import pandas as pd
from pandas import read_excel

batch=1000
def Nearest_Points(filt_df, medoids):
    distance=Calculate_Distance(filt_df[:,None,:], filt_df[None,medoids,:])
    return np.argmin(distance,axis=1)
    
def Calculate_Distance(a,b):
    return np.sum(np.abs(a - b), axis=-1)

def Finding_Medoids(filt_df,df_length,clust_num,assignments):
    medoid_ids = np.full(clust_num, -1, dtype=int)
    subset = np.random.choice(df_length, batch, replace=False)

    for i in range(clust_num):
        index = np.intersect1d(np.where(assignments==i)[0], subset)
        distance = Calculate_Distance(filt_df[index, None, :], filt_df[None, index, :]).sum(axis=0)
        medoid_ids[i] = index[np.argmin(distance)]

    return medoid_ids


def K_Meds(filt_df,df_length,clust_num,iterations=30):
    print("Initializing to random medoids.")
    medoids = np.random.choice(df_length, clust_num, replace=False)
    print("initial id of medoids",medoids)
    assignments = Nearest_Points(filt_df,medoids)

    for i in range(iterations):
        print("\tFinding new medoids.")
        medoids = Finding_Medoids(filt_df,df_length,clust_num,assignments)
        print("new id of medoids",medoids)
        print("\tReassigning points.")
        new_assignments = Nearest_Points(filt_df,medoids)

        difference = np.mean(new_assignments != assignments)
        assignments = new_assignments

        print("iteration {:2d}: {:.2%} of points got reassigned." "".format(i, difference))
        if difference <= 0.01:
            break
    #print(type(class_assignments))
    return assignments, medoids

def K_Medoids_Clustering(filt_df,clusters):
    df_length=len(filt_df)
    filt_df1=np.array(filt_df)
    clust_num=clusters
    final_assignments, final_medoid_ids = K_Meds(filt_df1,df_length,clust_num)
    print(final_assignments, final_medoid_ids)
  
    #return final_assignments, final_medoid_ids
    return final_assignments
