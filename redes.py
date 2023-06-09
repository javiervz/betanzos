# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:47:10 2023

@author: jveraz
"""

import re

##################################
#### parte 1: leer el archivo ####
##################################

## abrir archivo
## corregí las líneas del archivo!
texto = open('BETANZOS.txt', 'r', encoding='utf-8')

## leemos el texto
texto = texto.read()

## dividimos por saltos de línea!
texto = texto.split('\n')

## extraemos los números romanos
def romanos(s):
	return bool(re.search(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",s))

## extraemos los substring entre <>
def par(s):
    return re.search(r'\|(.*?)\|',s).group(1)

##########################################
#### parte 2: simplificamos los datos ####
##########################################

strings = []

for s in texto:
    try:
        s = s.replace(',','')
        tokens = s.split(' ')
        num = [t for t in tokens if romanos(t) and t!='']
        key = par(s)
        strings += [[key,num]]
    except AttributeError:
        print(s)
        pass
    
## join capítulos!
for i in range(len(strings)):
    s = strings[i]
    cap = ' '.join(s[1:][0])
    strings[i] = [s[0],cap]
    
## dict
D = dict(strings)
D = {'personas':list(D.keys()),'capítulos':list(D.values())}

## dataframe!
import pandas as pd

DF = pd.DataFrame.from_dict(D)
DF = DF.dropna()
###################################
#### parte 3: juegos con excel ####
###################################

## La primera columna funciona como keys, la segunda como values

personas = list(DF['personas'])
capitulos = list(DF['capítulos'])

D = {}
for i in range(len(personas)):
    p = personas[i]
    c = capitulos[i]
    
    D[p] = c

d = {}
## Modifique un poco el diccionario: los values son strings! Ahora deben ser listas de capítulos!
for p in personas:
    ## accedemos a los capítulos asociados a p. s es un string
    s = D[p]
    ## transformar s en lista
    L = s.split(' ')
    ## actualizamos d
    d[p] = L
        
###################################
## parte 4: funciones divertidas ##
###################################

## Defina una función F reciba dos keys cualquiera del diccionario de la parte 3. La función debe entregar
## 1 si los keys comparten al menos 1 capítulo, y 0 en otro caso. 

def F(p1,p2):
    ## accedemos a las listas
    l1 = d[p1]
    l2 = d[p2]
    
    ## intersección
    inter = 0
    ## comparamos las listas
    for elemento in l1:
        if elemento in l2:
            inter += 1
    return inter/len(set(l1+l2))

print(F('yawira', 'yuqa-y'))
       
## diccionario de diccionarios
D = {}
for key in d.keys():
    D[key] = {}

for key in d.keys():
    for keykey in d.keys():
            D[key][keykey] = F(key,keykey)
## para crear un diccionario de diccionarios 

#####################
## parte 5: redes! ##
#####################

## Usamos la librería networkx
import networkx as nx

## definimos una red sin elementos simples, ni conexiones
G = nx.Graph()

## recorremos D
for key in D.keys():
    for keykey in D.keys():
        if D[key][keykey] > 0.95 and key!=keykey:
            G.add_edge(key,keykey,weigth=F(key,keykey))
        
## podamos aristas
Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G = G.subgraph(Gcc[0])

#G = nx.maximum_spanning_tree(G, weight='weight')
    
##########################################
## parte 6: descripción y visualización ##
##########################################

print(len(G),len(G.edges()))

import matplotlib.pyplot as plt

pos = nx.kamada_kawai_layout(G)#, iterations=100, seed=0)
labels={i:i for i in G.nodes()}# if families[i] in ['Panoan','Arawakan','Mayan','Otomanguean','Quechuan']}
nx.draw_networkx_nodes(G, pos, node_size = 10, node_color='orange',linewidths=0.1,alpha=0.85) 
nx.draw_networkx_edges(G, pos, alpha=0.5,width=0.25,edge_color='gray')
#nx.draw_networkx_labels(G,pos,labels,alpha=0.95,font_size=6,font_color='k',font_family='monospace')
plt.axis('off')
plt.savefig('graph.jpg', format='jpg', transparent=True, bbox_inches='tight',dpi=1080)
plt.show()

#########################
## parte 7: clustering ##
#########################

from sklearn.cluster import KMeans
import numpy as np

# Obtiene la matriz de adyacencia del grafo
adj_matrix = np.asarray(nx.to_numpy_matrix(G))

# Realiza el clustering utilizando k-means
num_clusters = 2  # Número de clusters deseado
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(adj_matrix)

# Obtiene las etiquetas de cluster asignadas a cada nodo
cluster_labels = kmeans.labels_

# Imprime los resultados
C = dict(zip(G.nodes(), cluster_labels))

## centralidad
eigenvector_centrality = nx.betweenness_centrality(G)

## ordenamos la centralidad
centrality_dict_sorted = dict(sorted(eigenvector_centrality.items(), key=lambda item: item[1], reverse=True))

## nodos más importantes
C0 = list({n:centrality_dict_sorted[n] for n in centrality_dict_sorted.keys() if C[n] == 0}.keys())[:5]
C1 = list({n:centrality_dict_sorted[n] for n in centrality_dict_sorted.keys() if C[n] == 1}.keys())[:5]

## labels
labels0 = {n:n for n in C0}
labels1 = {n:n for n in C1}

labels0.update(labels1)

## visualizamos los clusters!
pos = nx.kamada_kawai_layout(G)#, iterations=100, seed=0)
nx.draw_networkx_nodes(G, pos, node_size = 10, node_color=['orange' if C[node] == 1 else 'cyan' for node in G.nodes()],linewidths=0.1,alpha=0.85) 
nx.draw_networkx_edges(G, pos, alpha=0.5,width=0.25,edge_color='gray')
nx.draw_networkx_labels(G,pos,labels0,alpha=0.65,font_size=4,font_color='k',font_family='monospace')
plt.axis('off')
plt.savefig('graph_clusters.jpg', format='jpg', transparent=True, bbox_inches='tight',dpi=1080)
plt.show()