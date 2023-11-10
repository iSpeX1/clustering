import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


###################################################################
# Exemple : Agglomerative Clustering


#######################################################################
### Données initiales
#######################################################################

path = './artificial/'
name="chainlink.arff"

#R15 good pour clustering agglomératif

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


#######################################################################
### Fixer la distance
#######################################################################


# Iteration du clustering agglomératif sur distance_threshold, évaluation avec silhouette
def silhouette_agglo_linkage(linkage_arg):
    silhouettes = []
    #The range start at 1 because the distance_threshold must be at least 1
    for seuil_dist in range(1,11):
        tps1 = time.time()
        model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage=linkage_arg, n_clusters=None)
        model = model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_
        leaves=model.n_leaves_
        k = model.n_clusters_
        silhouettes.append(metrics.silhouette_score(datanp, labels, metric='euclidean'))
        print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")

    print("Silhouettes : ")
    print(silhouettes)

    # Plot the evolution of the silhouettes
    plt.figure(figsize=(6, 6))
    plt.plot(range(1,11),silhouettes)
    plt.title("Evolution de la silhouette avec linkage = "+str(linkage_arg))
    #plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-silhouette.jpg",bbox_inches='tight', pad_inches=0.1)
    plt.show()

#APPELS DES FONCTIONS AVEC LES DIFFERENTS LINKAGE
#silhouette_agglo_linkage('single')
#silhouette_agglo_linkage('complete')
#silhouette_agglo_linkage('average')
#silhouette_agglo_linkage('ward')



#plt.scatter(f0, f1, c=labels, s=8)
#plt.title("Clustering agglomératif (average, distance_treshold= "+str(seuil_dist)+") "+str(name))
#plt.show()


#######################################################################
# Fixer le nombre de clusters
#######################################################################

# Iteration du clustering agglomératif sur le nombre de clusters, évaluation avec silhouette

def silhouette_agglo_nclusters(linkage_arg, show_data):
    silhouettes = []
    #The range start at 2 because the number of clusters must be at least 2
    for k in range(2,11):
        tps1 = time.time()
        model = cluster.AgglomerativeClustering(linkage=linkage_arg, n_clusters=k)
        model = model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_
        leaves=model.n_leaves_
        silhouettes.append(metrics.silhouette_score(datanp, labels, metric='euclidean'))
        print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")

    print("Silhouettes : ")
    print(silhouettes)

    # Plot the evolution of the silhouettes
    plt.figure(figsize=(6, 6))
    plt.plot(range(2,11),silhouettes)
    plt.title("Evolution de la silhouette avec linkage = "+str(linkage_arg))
    #plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-silhouette.jpg",bbox_inches='tight', pad_inches=0.1)
    plt.show()
    if show_data:
        max_silhouette = np.argmax(silhouettes)+2
        model = cluster.AgglomerativeClustering(linkage=linkage_arg, n_clusters=max_silhouette)
        model = model.fit(datanp)
        labels = model.labels_
        plt.scatter(f0, f1, c=labels, s=8)
        plt.title("Clustering agglomératif ("+ str(linkage_arg)+", n_cluster= "+str(max_silhouette)+") "+str(name))
        plt.show()


#APPELS DES FONCTIONS AVEC LES DIFFERENTS LINKAGE
show=True
silhouette_agglo_nclusters('single',show)
silhouette_agglo_nclusters('complete',show)
silhouette_agglo_nclusters('average',show)
silhouette_agglo_nclusters('ward',show)




#######################################################################

