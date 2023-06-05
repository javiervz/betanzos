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
#DF.to_excel('datos.xlsx')
DF = DF.dropna()
###################################
#### parte 3: juegos con excel ####
###################################

## Lea el archivo excel datos.xlsx. Use pandas
## ¿Cuáles son las columnas?
## Transforme en diccionario el dataframe. La primera columna funciona como keys, la segunda como values

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
    return inter

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
        if F(key,keykey)>1:
            G.add_edge(key,keykey,weigth=F(key,keykey))
## podamos aristas
Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G = G.subgraph(Gcc[0])

G = nx.maximum_spanning_tree(G, weight='weight')
    
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

from node2vec import Node2Vec
from sklearn.decomposition import PCA
import numpy as np

# Crea un objeto Node2Vec y entrena el modelo
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(vector_size = 16, window=10, min_count=1)

# Obtiene los vectores de características para cada nodo en forma de diccionario
embeddings = {node: model.wv[node] for node in G.nodes()}

# Convierte los vectores de características en una matriz numpy
node_ids = list(embeddings.keys())
node_vectors = np.array(list(embeddings.values()))

# Aplica PCA para reducir la dimensionalidad de los vectores a 2 dimensiones
pca = PCA(n_components=2)
node_vectors_2d = pca.fit_transform(node_vectors)

# Visualiza los nodos en el espacio reducido
plt.figure(figsize=(8, 6))
plt.scatter(node_vectors_2d[:, 0], node_vectors_2d[:, 1], c='b')
for i, node_id in enumerate(node_ids):
    plt.annotate(node_id, (node_vectors_2d[i, 0], node_vectors_2d[i, 1]))
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualización de nodos con PCA')
plt.show()
