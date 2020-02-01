import numpy as np
import pandas as pd
from pandas import read_excel

######################### K-Medoids

def assign_nearest(ids_of_mediods):
    dists = dist(x[:,None,:], x[None,ids_of_mediods,:])
    return np.argmin(dists, axis=1)


def dist(xa, xb):
    if EUCLIDEAN:
        return np.sqrt(np.sum(np.square(xa-xb), axis=-1))
    else:
        return np.sum(np.abs(xa - xb), axis=-1)


def find_medoids(assignments):
    medoid_ids = np.full(k, -1, dtype=int)
    subset = np.random.choice(n, batch_sz, replace=False)

    for i in range(k):
        indices = np.intersect1d(np.where(assignments==i)[0], subset)
        distances = dist(x[indices, None, :], x[None, indices, :]).sum(axis=0)
        medoid_ids[i] = indices[np.argmin(distances)]

    return medoid_ids


def kmeds(iterations=20):
    print("Initializing to random medoids.")
    ids_of_medoids = np.random.choice(n, k, replace=False)
    class_assignments = assign_nearest(ids_of_medoids)

    for i in range(iterations):
        print("\tFinding new medoids.")
        ids_of_medoids = find_medoids(class_assignments)
        print("\tReassigning points.")
        new_class_assignments = assign_nearest(ids_of_medoids)

        diffs = np.mean(new_class_assignments != class_assignments)
        class_assignments = new_class_assignments

        print("iteration {:2d}: {:.2%} of points got reassigned." "".format(i, diffs))
        if diffs <= 0.01:
            break

    return class_assignments, ids_of_medoids

def fox(clusters):
    print("HELLO")
    df=pd.read_excel(r'C:\\Users\\agarw\\source\\repos\\WebProject1\\WebProject1\\Database_Objects.xlsx')
    print(df.head())
    x=df.iloc[:,5:]
    print(x.head())
    x=np.array(x)
    print(x[:5,:2])      
    print("YES")
    batch_sz = 1000
    n=len(x)
    k=clusters
    #K_medoids_func()
    EUCLIDEAN = False

    #print("n={}\td={}\tk={}\tbatch_size={} ".format(n, d, k, batch_sz))
    #print("Distance metric: ", "Eucledian" if EUCLIDEAN else "Manhattan")
    final_assignments, final_medoid_ids = kmeds()
    print(final_assignments, final_medoid_ids)
    #return final_assignments, final_medoid_ids
