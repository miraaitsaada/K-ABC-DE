#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pylab
import numpy as np
import K_ABC_DE
import random
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import time

def clustering_KABCDE (nom_fichier , delimiteur , k):
    
    data = np.loadtxt(nom_fichier, delimiter=delimiteur)
    
    debut = time.time()
    resultat = K_ABC_DE.K_ABC_DE (data, k)
    fin = time.time()
    
    temps = fin-debut
    
    SSE = K_ABC_DE.SSE(data, k, resultat)
    
    colormap = np.array(['#F06A73','#46C6AC','#919191'])
    
    data_size = (data).shape[0]
    d=np.size(data[0])
    
    classification = np.empty(data_size, dtype=int)
    
    for indice_donnee in range(data_size):
        indice_centroid = min([(i[0], np.linalg.norm(data[indice_donnee]-resultat[i[0]])) \
                                for i in enumerate(resultat)], key=lambda t:t[1])[0]
        classification[indice_donnee] = indice_centroid
    
    nb_elements = np.zeros(k, dtype = int)
    moyenne_elements = np.zeros((k,d))
    elements = [[]]
    for i in range(k-1):
        elements.append([])
    
    for i, j in enumerate(classification):
        for g in range(k):
            if(g == j):
                elements[g].append(data[i].tolist())
                moyenne_elements[j] = np.add(moyenne_elements[j] , data[i])
                nb_elements[j] +=1
                
    for i in range(k):
        moyenne_elements[i] = moyenne_elements[i] / np.copy(nb_elements[i])
    
    resultat_dans_lordre = np.empty((k,d))
    
    indice_deja_pris = []
    
    for i, centroid in enumerate(moyenne_elements):
        indice_le_plus_proche = 0
        while(indice_le_plus_proche in indice_deja_pris):
            indice_le_plus_proche += 1
            
        distance_minimale = np.linalg.norm(centroid - resultat[indice_le_plus_proche])
        
        for j, cluster_y in enumerate(moyenne_elements):
            if(not (j in indice_deja_pris)):
                distance = np.linalg.norm(centroid - resultat[j])
                if(distance <= distance_minimale):
                     distance_minimale = distance
                     indice_le_plus_proche = j
    
        indice_deja_pris.append(indice_le_plus_proche)
        resultat_dans_lordre[i] = resultat[indice_le_plus_proche]
    
    resultat = np.copy(resultat_dans_lordre)
    
    if (d==2):
        
        gs = gridspec.GridSpec(7, 4)
    
        fig = pylab.figure(1, figsize=(5,10))
        
        plt.title('Classification à 2 dimensions du fichier : " '+nom_fichier+' "\n' ,fontsize = 8)
    
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        
        ax = fig.add_subplot(gs[0,0])
        ax.axis('off')
        affichage = "SSE = " + ("{0:.4f}".format(SSE)) + "\n" + "Nombre de clusters : " + str(k) + "\n" + "Nombre d'attributs : " + str(d) + "\n"
        affichage = affichage  + "Temps de classification : " + ("{0:.1f}".format(temps)) + " s"
        
        plt.text(0, 1,affichage,
             horizontalalignment='left',
             verticalalignment='top',
             transform = ax.transAxes,
             fontsize = 8)
        
        ax = fig.add_subplot(gs[0,2:])
        ax.axis('off')
        st = ''
        for i in range(k):
            cluster = np.empty(d, dtype=float)
            for x,y  in enumerate(resultat[i]):
                cluster[x] = ("{0:.3f}".format(y))
            clust = ''
            for j in range(d):    
                clust = clust + str(cluster[j]) + ' ; '
            st = st + "Cluster " + str(i+1) + ' : [ ' + clust[:-2]  + ']' + "\n"
                
        affichage = st
        plt.text(0, 1,affichage,
             horizontalalignment='left',
             verticalalignment='top',
             transform = ax.transAxes,
             fontsize = 8)
        
        ax = fig.add_subplot(gs[1:,:])
        
        plt.tick_params(direction='out', length=2, width=1, labelsize = '10')
        clusts = []
        for i in range(k):
            clusts.append(plt.scatter(np.array(elements[i]).T[0], np.array(elements[i]).T[1], c=colormap[i],  marker='o', s=40, alpha = 0.4, label = "Cluster " + str(i+1) + ' (' + str(nb_elements[i]) + ' individus)' ) )
    
        ax.scatter(moyenne_elements.T[0], moyenne_elements.T[1], s= 55, marker = 'x', color = 'black')
        
        plt.legend(handles=clusts, prop={'size':7})#
        plt.show()
    
    else:
        if(d==4):
            gs = gridspec.GridSpec(7, 4)
    
            fig = pylab.figure(1, figsize=(5,10))
            
            plt.title('Représentation 3D du clustering du fichier : " '+nom_fichier+' "\n' ,fontsize = 8)
    
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
                    
            ax = fig.add_subplot(gs[0,0])
            ax.axis('off')
            affichage = "SSE = " + ("{0:.4f}".format(SSE)) + "\n" + "Nombre de clusters : " + str(k) + "\n" + "Nombre d'attributs : " + str(d)
            affichage = affichage + "\n" + "Temps de classification : " + ("{0:.1f}".format(temps)) + " s"
        
            plt.text(0, 1,affichage,
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform = ax.transAxes,
                 fontsize = 8)
            
            ax = fig.add_subplot(gs[0,2:])
            
            ax.axis('off')
            st = ''
            for i in range(k):
                cluster = np.empty(d, dtype=float)
                for x,y  in enumerate(resultat[i]):
                    cluster[x] = ("{0:.3f}".format(y))
                clust = ''
                for j in range(d):    
                    clust = clust + str(cluster[j]) + ' ; '
                st = st + "Cluster " + str(i+1) + ' : [ ' + clust[:-2]  + ']' + "\n"
                    
            affichage = st
            plt.text(0, 1,affichage,
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform = ax.transAxes,
                 fontsize = 8)
            
            ax = fig.add_subplot(gs[1:4,:2], projection='3d')
            clusts = []
            for i in range(k):
                clusts.append (ax.scatter(np.array(elements[i]).T[1], np.array(elements[i]).T[2], np.array(elements[i]).T[0], c=colormap[i], marker='o', s=25, alpha = 0.4, label = "Cluster " + str(i+1) + ' (' + str(nb_elements[i]) + ' individus)') )
                
            ax.scatter(moyenne_elements.T[1], moyenne_elements.T[2], moyenne_elements.T[0] , s= 55, marker = 'x', color = 'black')
     
            ax.set_xlabel('(2)', fontsize = 6)
            ax.set_ylabel('(3)', fontsize = 6)
            ax.set_zlabel('(1)',fontsize = 6)
            plt.tick_params(axis = "x" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
            plt.tick_params(axis = "z" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
            plt.tick_params(axis = "y" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
            
            ax = fig.add_subplot(gs[1:4,2:], projection='3d')
            clusts = []
            for i in range(k):
                clusts.append (ax.scatter(np.array(elements[i]).T[3], np.array(elements[i]).T[1], np.array(elements[i]).T[0], c=colormap[i], marker='o', s=25, alpha = 0.4, label = "Cluster " + str(i+1) + ' (' + str(nb_elements[i]) + ' individus)') )
                
            ax.scatter(moyenne_elements.T[3], moyenne_elements.T[1], moyenne_elements.T[0] , s= 55, marker = 'x', color = 'black')
    
            ax.set_xlabel('(4)', fontsize = 6) 
            ax.set_ylabel('(2)', fontsize = 6) 
            ax.set_zlabel('(1)', fontsize = 6)
            plt.tick_params(axis = "x" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
            plt.tick_params(axis = "z" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
            plt.tick_params(axis = "y" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
            
            plt.legend(handles=clusts, prop={'size':7})#
            
            ax = fig.add_subplot(gs[4:7,:2], projection='3d')
            clusts = []
            for i in range(k):
                clusts.append (ax.scatter(np.array(elements[i]).T[0], np.array(elements[i]).T[2], np.array(elements[i]).T[3], c=colormap[i], marker='o', s=25, alpha = 0.4, label = "Cluster " + str(i+1) + ' (' + str(nb_elements[i]) + ' individus)') )
                
            ax.scatter(moyenne_elements.T[0], moyenne_elements.T[2], moyenne_elements.T[3] , s= 55, marker = 'x', color = 'black')
    
            ax.set_xlabel('(1)', fontsize = 6) 
            ax.set_ylabel('(3)', fontsize = 6) 
            ax.set_zlabel('(4)', fontsize = 6)
            plt.tick_params(axis = "x" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
            plt.tick_params(axis = "z" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
            plt.tick_params(axis = "y" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
            
            ax = fig.add_subplot(gs[4:7,2:], projection='3d')
            
            clusts = []
            for i in range(k):
                clusts.append (ax.scatter(np.array(elements[i]).T[1], np.array(elements[i]).T[3], np.array(elements[i]).T[2], c=colormap[i], marker='o', s=25, alpha = 0.4, label = "Cluster " + str(i+1) + ' (' + str(nb_elements[i]) + ' individus)') )
                
            ax.scatter(moyenne_elements.T[1], moyenne_elements.T[3], moyenne_elements.T[2] , s= 55, marker = 'x', color = 'black')
    
            ax.set_xlabel('(2)', fontsize = 6) 
            ax.set_ylabel('(4)', fontsize = 6) 
            ax.set_zlabel('(3)', fontsize = 6)
            plt.tick_params(axis = "x" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
            plt.tick_params(axis = "z" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
            plt.tick_params(axis = "y" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
            plt.show()
                           
        else :
            if(d==3):
                gs = gridspec.GridSpec(7, 4)
    
                fig = pylab.figure(1, figsize=(5,10))
                plt.title('Représentation 3D du clustering du fichier : " '+nom_fichier+' "\n' ,fontsize = 8)
    
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
                
                frame = plt.gca()
                frame.axes.get_xaxis().set_visible(False)
                frame.axes.get_yaxis().set_visible(False)
                
                ax = fig.add_subplot(gs[0,0])
                ax.axis('off')
                affichage = "SSE = " + ("{0:.4f}".format(SSE)) + "\n" + "Nombre de clusters : " + str(k) + "\n" + "Nombre d'attributs : " + str(d)
                affichage = affichage + "\n" + "Temps de classification : " + ("{0:.1f}".format(temps)) + " s"
        
                plt.text(0, 1,affichage,
                     horizontalalignment='left',
                     verticalalignment='top',
                     transform = ax.transAxes,
                     fontsize = 8)
                
                ax = fig.add_subplot(gs[0,2:])
                
                ax.axis('off')
                st = ''
                for i in range(k):
                    cluster = np.empty(d, dtype=float)
                    for x,y  in enumerate(resultat[i]):
                        cluster[x] = ("{0:.3f}".format(y))
                    clust = ''
                    for j in range(d):    
                        clust = clust + str(cluster[j]) + ' ; '
                    st = st + "Cluster " + str(i+1) + ' : [ ' + clust[:-2]  + ']' + "\n"
                        
                affichage = st
                plt.text(0, 1,affichage,
                     horizontalalignment='left',
                     verticalalignment='top',
                     transform = ax.transAxes,
                     fontsize = 8)
                
                ax = fig.add_subplot(gs[1:,:], projection='3d')
                
                clusts = []
                for i in range(k):
                    clusts.append (ax.scatter(np.array(elements[i]).T[0], np.array(elements[i]).T[1], np.array(elements[i]).T[2], c=colormap[i], marker='o', s=40, alpha = 0.4, label = "Cluster " + str(i+1) + ' (' + str(nb_elements[i]) + ' individus)') )
                    
                ax.scatter(moyenne_elements.T[0], moyenne_elements.T[1], moyenne_elements.T[2] , s= 55, marker = 'x', color = 'black')
        
                plt.tick_params(axis = "x" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
                plt.tick_params(axis = "z" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
                plt.tick_params(axis = "y" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
    
                plt.legend(handles=clusts, prop={'size':7})#
                plt.show()
        
            else :
                if(d>4):
                    gs = gridspec.GridSpec(7, 4)
    
                    fig = pylab.figure(1, figsize=(5,10))
                    plt.title('Représentation 3D du clustering du fichier : " '+nom_fichier+' "\n' ,fontsize = 8)
    
                    for spine in plt.gca().spines.values():
                        spine.set_visible(False)
                    
                    frame = plt.gca()
                    frame.axes.get_xaxis().set_visible(False)
                    frame.axes.get_yaxis().set_visible(False)
    
                    ax = fig.add_subplot(gs[0,0])
                    
                    ax.axis('off')
                    affichage = "SSE = " + ("{0:.4f}".format(SSE)) + "\n" + "Nombre de clusters : " + str(k) + "\n" + "Nombre d'attributs : " + str(d)
                    affichage = affichage + "\n" + "Temps de classification : " + ("{0:.1f}".format(temps)) + " s"
        
                    plt.text(0, 1,affichage,
                         horizontalalignment='left',
                         verticalalignment='top',
                         transform = ax.transAxes,
                         fontsize = 8)
                    
                    ax = fig.add_subplot(gs[0,2:])
                    
                    ax.axis('off')
                    st = ''
                    for i in range(k):
                        cluster = np.empty(d, dtype=float)
                        for x,y  in enumerate(resultat[i]):
                            cluster[x] = ("{0:.3f}".format(y))
                        clust = ''
                        for j in range(d):    
                            clust = clust + str(cluster[j]) + ' ; '
                        st = st + "Cluster " + str(i+1) + ' : [ ' + clust[:-2]  + ']' + "\n"
                            
                    affichage = st
                    plt.text(0, 1,affichage,
                         horizontalalignment='left',
                         verticalalignment='top',
                         transform = ax.transAxes,
                         fontsize = 8)
            
                    ax = fig.add_subplot(gs[1:4,:2], projection='3d')
                    
                    indice1, indice2, indice3 = random.sample(range(0, d-1), 3)
       
                    clusts = []
                    for i in range(k):
                        clusts.append (ax.scatter(np.array(elements[i]).T[indice1], np.array(elements[i]).T[indice2], np.array(elements[i]).T[indice3], c=colormap[i], marker='o', s=25, alpha = 0.4, label = "Cluster " + str(i+1) + ' (' + str(nb_elements[i]) + ' individus)') )
                        
                    ax.scatter(moyenne_elements.T[indice1], moyenne_elements.T[indice2], moyenne_elements.T[indice3] , s= 55, marker = 'x', color = 'black')
    
                    ax.set_xlabel('('+str(indice1+1)+')', fontsize = 6)
                    ax.set_ylabel('('+str(indice2+1)+')', fontsize = 6)
                    ax.set_zlabel('('+str(indice3+1)+')',fontsize = 6)
                    plt.tick_params(axis = "x" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
                    plt.tick_params(axis = "z" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
                    plt.tick_params(axis = "y" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
                    
                    ax = fig.add_subplot(gs[1:4,2:], projection='3d')
                    
                    plt.legend(handles=clusts, prop={'size':7})#
                    
                    indice1, indice2, indice3 = random.sample(range(0, d-1), 3)
                    clusts = []
                    for i in range(k):
                        clusts.append (ax.scatter(np.array(elements[i]).T[indice1], np.array(elements[i]).T[indice2], np.array(elements[i]).T[indice3], c=colormap[i], marker='o', s=25, alpha = 0.4, label = "Cluster " + str(i+1) + ' (' + str(nb_elements[i]) + ' individus)') )
                        
                    ax.scatter(moyenne_elements.T[indice1], moyenne_elements.T[indice2], moyenne_elements.T[indice3] , s= 55, marker = 'x', color = 'black')
    
                    ax.set_xlabel('('+str(indice1+1)+')', fontsize = 6)
                    ax.set_ylabel('('+str(indice2+1)+')', fontsize = 6)
                    ax.set_zlabel('('+str(indice3+1)+')',fontsize = 6)
                    plt.tick_params(axis = "x" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
                    plt.tick_params(axis = "z" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
                    plt.tick_params(axis = "y" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
                    
                                   
                    ax = fig.add_subplot(gs[4:7,:2], projection='3d')
                    indice1, indice2, indice3 = random.sample(range(0, d-1), 3)
                    clusts = []
                    for i in range(k):
                        clusts.append (ax.scatter(np.array(elements[i]).T[indice1], np.array(elements[i]).T[indice2], np.array(elements[i]).T[indice3], c=colormap[i], marker='o', s=25, alpha = 0.4, label = "Cluster " + str(i+1) + ' (' + str(nb_elements[i]) + ' individus)') )
                        
                    ax.scatter(moyenne_elements.T[indice1], moyenne_elements.T[indice2], moyenne_elements.T[indice3] , s= 55, marker = 'x', color = 'black')
    
                    ax.set_xlabel('('+str(indice1+1)+')', fontsize = 6)
                    ax.set_ylabel('('+str(indice2+1)+')', fontsize = 6)
                    ax.set_zlabel('('+str(indice3+1)+')',fontsize = 6)
                    plt.tick_params(axis = "x" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
                    plt.tick_params(axis = "z" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
                    plt.tick_params(axis = "y" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
                    
                    
                    ax = fig.add_subplot(gs[4:7,2:], projection='3d')
                    indice1, indice2, indice3 = random.sample(range(0, d-1), 3)
                    clusts = []
                    for i in range(k):
                        clusts.append (ax.scatter(np.array(elements[i]).T[indice1], np.array(elements[i]).T[indice2], np.array(elements[i]).T[indice3], c=colormap[i], marker='o', s=25, alpha = 0.4, label = "Cluster " + str(i+1) + ' (' + str(nb_elements[i]) + ' individus)') )
                        
                    ax.scatter(moyenne_elements.T[indice1], moyenne_elements.T[indice2], moyenne_elements.T[indice3] , s= 55, marker = 'x', color = 'black')
    
                    ax.set_xlabel('('+str(indice1+1)+')', fontsize = 6)
                    ax.set_ylabel('('+str(indice2+1)+')', fontsize = 6)
                    ax.set_zlabel('('+str(indice3+1)+')',fontsize = 6)
                    plt.tick_params(axis = "x" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
                    plt.tick_params(axis = "z" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
                    plt.tick_params(axis = "y" , pad = 0.1, labelsize = '6')#direction='in', length=1.5, width=1,
                    plt.show()
            
if __name__ == "__main__":
    
    #nom_fichier = 'iris.data.txt'
    nom_fichier = "iris3D.data.txt"
    #nom_fichier = "iris_sepal.data.txt"
    #nom_fichier = "iris_petal.data.txt"
    
    delimiteur = ','
    
    k=3
    
    clustering_KABCDE(nom_fichier , ',' , 3)
    
    
