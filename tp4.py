#tp4



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import random as rd
import seaborn as sns



#exercice 1

def ClusterRandomGraph(p, q, k=4, n=100): #The graph has 400 nodes partition into 4 clusters of 100 nodes.

    edges = []
    noeuds = np.arange(k*n)  
    for i,n_1 in enumerate(noeuds):
        cluster_actuel = i // n
        for n_2 in noeuds[i+1:(cluster_actuel+1)*n]:  #les aretes ne sont pas comptées 2 fois
            if np.random.random_sample() < p:  # Each pair of nodes in the same cluster is connected with a probability p
                edges.append((n_1, n_2))  
        for n_3 in noeuds[(cluster_actuel+1)*n:]:  #les aretes ne sont pas comptées 2 fois
            if np.random.random_sample() < q:  #each pair of nodes in different clusters is connected with a probability q<=p
                edges.append((n_1, n_3))
    return edges



#exercice 2

# Program to shuffle a given array using Fisher–Yates shuffle Algorithm


#on doit d'abord convertir la liste des aretes en matrice d'adjacence

def conv_list_of_edges_adj_array(G):
    G_nx = nx.from_edgelist(G)  
    arr = nx.to_dict_of_dicts(G_nx) 
    for idx,v in arr.items():
        arr[idx] = [i for i in v.keys()]
    return arr


# A function to generate a random permutation of arr[]
def randomize (arr):
    n=len(arr)
    # Start from the last element and swap one by one. We don't
    # need to run for the first element that's why i > 0
    for i in range(n-1,0,-1):
        # Pick a random index from 0 to i
        j = rd.randint(0,i+1)
  
        # Swap arr[i] with the element at random index
        arr[i],arr[j] = arr[j],arr[i]
    return arr



def label_propagation(arr):
    #renvoie une liste des labels de chaque noeuds
    label = np.arange(len(arr))  
    iter = list(arr.keys())  
    change = 1 #indicateur de changement a chaque iteration
    while change> 0: #tant que on a du changement
        label_prec = label.copy()  
        change = 0  
        iter = randomize(iter)  #on melange la liste
        for i in iter:  #on fait ca pour chaque noeud
            label_voisins = [label[voisin] for voisin in arr[i]]  #on recupere les voisins du noeud i
            label[i] = max(label_voisins, key=label_voisins.count)  # label avec la plus haute fréquence
            change+= abs(label[i] - label_prec[i])  #ajoute un terme non nul si changement 
    return label






#Affichage coloré du graphe
def graphs(arr, label):    
    
    arr_nx = nx.from_dict_of_lists(arr) #conversion en nx
    
    labels=set(label) #on recupere les labels possibles du graphe
    
    indices = np.arange(len(labels))
    
    label_dict = dict(zip(labels, indices)) # on cree un dict qui lie chaque label à un indice, pour pouvoir attribuer la meme couleur à chaque label
    colors = sns.color_palette(None, len(labels))  
    
    c = []  
    for noeud in arr_nx:
        indice = label_dict[label[noeud]]
        c.append(colors[indice])
    
    nx.draw(arr_nx, node_color=c, with_labels=False)
    plt.show()

#exercice 1
#On utilise la librairie NetworkX et matlplotlib pour dessiner le graphe

G = ClusterRandomGraph(0.7, 0.01)
#G_nx = nx.from_edgelist(G)
#nx.draw(G_nx, with_labels=False)
#plt.show()


#exercice 2

#test des fonctions

a=conv_list_of_edges_adj_array(G)
label=label_propagation(a)
graphs(a,label) #on obtient bien un graphe avec une couleur par cluster


