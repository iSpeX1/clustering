"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="xclara.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
tps1 = time.time()

def good_inerties():
    # pour k qui va de 1 à 10
    inerties = []
    for k in range(1,11):
        #kmeans
        #model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
        #MiniBatch
        model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)   
        model.fit(datanp)
        inertie = model.inertia_
        inerties.append(inertie)
    print("Inerties : ")
    print(inerties)

    # Plot the evolution of the inerties
    plt.figure(figsize=(6, 6))
    plt.plot(range(1,11),inerties)
    plt.title("Evolution de l'inertie intra-classe")
    #plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-inerties.jpg",bbox_inches='tight', pad_inches=0.1)
    plt.show()

    #calcul du changement de pente
    print("Changement de pente : ")
    print(np.diff(inerties))

    # find where the difference is the largest
    print("Indice du changement de pente : ")
    print(np.argmax(np.diff(inerties))+2)

    print("On peut constater graphiquement un changement drastic de pente à k=3 (jeux de données xclara)")
    print("On peut donc en déduire que le nombre de clusters optimal est 3")

    return 3

def silhouette():
    # pour k qui va de 2 à 10
    silhouettes = []
    for k in range(2,11):
        #kmeans
        #model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
        #MiniBatch
        model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
        model.fit(datanp)
        labels = model.labels_
        silhouette = metrics.silhouette_score(datanp, labels, metric='euclidean')
        silhouettes.append(silhouette)
    
    #print silhouette table
    print("Silhouettes : ")
    print(silhouettes)

    # Plot the evolution of the silhouettes
    plt.figure(figsize=(6, 6))
    plt.plot(range(2,11),silhouettes)
    plt.title("Evolution de la silhouette")
    #plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-silhouette.jpg",bbox_inches='tight', pad_inches=0.1)
    plt.show()

    return np.argmax(silhouettes)+2

# Optimal inertie
k = good_inerties()

# Optimal silhouette
#k = silhouette()
#print ("k optimal = ", k)

#Kmneans
#model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)

#MiniBatch
model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)

model.fit(datanp)

tps2 = time.time()

tps = tps2 - tps1
print("Temps de calcul : ", round(tps,2),"s")

labels = model.labels_

# informations sur le clustering obtenu
inertie = model.inertia_
iteration = model.n_iter_
centroids = model.cluster_centers_

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels)

####################### SEPARATION #######################

from sklearn.metrics.pairwise import euclidean_distances
dists = euclidean_distances(centroids)

print("------------------------------------------------------")
print("SEPARATION : ")

print("Distance min : ")
print(np.min(dists[dists>0]))

print("Distance max : ")
print(np.max(dists))

print("Distance moyenne : ")
print(np.mean(dists[dists>0]))

################# REGROUPEMENT #################
print("------------------------------------------------------")
print("REGROUPEMENT : ")

def k_means_distance(data,cx, cy, i_centroid, cluster_labels):
    distances = [np.sqrt((x-cx)**2+(y-cy)**2) for (x, y) in data[cluster_labels == i_centroid]]
    return distances

distances = []
for i, (cx, cy) in enumerate(centroids):
    mean_distance = k_means_distance(datanp,cx, cy, i, labels)
    distances.append(mean_distance)

min_distance = []
for i in range(len(distances)):
    min_distance.append(np.min(distances[i]))
print("Distances min : ")
print(min_distance)

max_distance = []
for i in range(len(distances)):
    max_distance.append(np.max(distances[i]))
print("Distances max : ")
print(max_distance)

mean_distance = []
for i in range(len(distances)):
    mean_distance.append(np.mean(distances[i]))
print("Distances moyennes : ")
print(mean_distance)