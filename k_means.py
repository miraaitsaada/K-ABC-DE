import numpy as np
import random
import matplotlib.pyplot as plt
#initialiser les individus (aléatoire ou pas)

#initialiser aléatoirement les centroïdes
def init_centroid (data, k):
    centroids = []
    for clust in range(0, k):
        centroids.append(data[random.randint(0, len(data)-1)])
    return centroids

#affecter chaque individus au cluster i au centroid le plus proche
def const_clusters (data, centroids, k):
    
    clusters = [[] for i in range(k)]
    
    for individu in data:
        mu_index=min([(i[0], np.linalg.norm(np.asarray(individu)-np.asarray(centroids[i[0]]))) \
                            for i in enumerate(centroids)], key=lambda t:t[1])[0]
    
    
        try:
            clusters[mu_index].append(individu)
        except KeyError:
            clusters[mu_index] = [individu]
    
    #si un cluster est vide alors lui affecter un élément aléatoirement
    for clust in clusters:
        if not clust:
            clust.append(data[random.randint(0, len(data))])

    return clusters


#recalculer les centroids (moyennes)
def recalcul_centroids (clusters, k):
    
    centroids = [[] for i in range(k)]
    
    index = 0
    for clust in clusters:
        centroids[index] = np.mean(clust, axis=0).tolist()
        index+=1
        
    return centroids

#verifier si les centroids ne changement pas
def has_converged (old_centroids , centroids , iterations):
    MAX_ITERATIONS = 1000
    if iterations > MAX_ITERATIONS :
        return True
    return old_centroids == centroids
    #return (set([tuple(a) for a in centroids]) == set([tuple(a) for a in old_centroids])


def k_means (data, k):

    centroids = init_centroid (data, k)
    old_centroids = []
    iterations = 0
    
    while not (has_converged (old_centroids , centroids, iterations)):
    
        iterations += 1
    
        clusters = const_clusters (data, centroids, k)
                        
        old_centroids = centroids
    
        centroids = recalcul_centroids (clusters, k)
        
#    print("Le nombre d'individus est : " + str(len(data)))
#    print("Le nombre d'itérations effectuées est : " + str(iterations))
#    print("Les centroïdes sont : " + str(centroids))
#    print("The clusters are as follows:")
#    for clust in clusters:
#        print("Cluster with a size of " + str(len(clust)) + " starts here:")
#        print(np.array(clust).tolist())
#        print("Cluster ends here.")

    return centroids
    



#data=[[1,2,3],[7,0,5],[99,55,6],[3,32,32],[3,32,31],[7,1,5],[1,1,3],[0,5,1]]
#data = np.load('clusterable_data.npy').tolist()
#
#
#
#
#
#start_time = time.time()
#clusters = k_means(data,2)
#end_time = time.time()
#
#
#
#data = np.array(data)
#clusters = np.array(clusters)
#d=np.size(data[0])
#data_size = (data).shape[0]
#
#
##affectation des clusters
#classification = np.empty(data_size, dtype=int)
#
#for indice_donnee in range(data_size):
#    indice_centroid = min([(i[0], np.linalg.norm(data[indice_donnee]-clusters[i[0]])) \
#                            for i in enumerate(clusters)], key=lambda t:t[1])[0]
#    classification[indice_donnee] = indice_centroid
#
#palette = sns.color_palette('deep', classification.max() + 1)
#colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in classification]
#plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
#frame = plt.gca()
#frame.axes.get_xaxis().set_visible(False)
#frame.axes.get_yaxis().set_visible(False)
#plt.title('Clusters found by k-means', fontsize=24)
#plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
#
#
#                  
#    
#
#


















