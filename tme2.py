#!/usr/bin/env python
# coding: utf-8

# ### kcore decomposition
# 
# In this notebook, you will find the python code for kcore decomposition (tme2). You can load any graph using the first function, which turns a txt file (usually a graph presented in list of edges) into a dictionnary structure of graph, which corresponds to the adjacency array.

# In[1]:


def load_graph (input):
    file = open(input,"r")
    file = file.readlines()
    G = dict()
    for line in file:
        try:
            s,t = map(int,line.split())
            try:
                G[s].append(t)
            except:
                G[s]=[t]
            try:
                G[t].append(s)
            except:
                G[t]=[s]
        except:
            pass # pour passer quand la ligne n'est pas au bon format
    return G

def load_graph2 (input): # Same, but we load the graph with its number of edges, which is way easier to compute using the
                         # initial data 
    file = open(input,"r")
    file = file.readlines()
    G = dict()
    nb_edges = 0
    for line in file:
        try:
            s,t = map(int,line.split())
            try:
                G[s].append(t)
                nb_edges+=1
            except:
                G[s]=[t]
            try:
                G[t].append(s)
            except:
                G[t]=[s]
        except:
            pass # pour passer quand la ligne n'est pas au bon format
    return (G, nb_edges)


# In[43]:


G, nb_edges = load_graph2('net.txt')
#G, nb_edges = load_graph2('com-amazon.ungraph.txt') # Loading one of the suggested graphs, e.g. the amazon network
#G, nb_edges = load_graph2('com-jl.ungraph.txt') 


# In[44]:


def core_decomposition(G):
    vertices = list(G.keys())
    edges = G
    i = len(list(vertices)) # Number of nodes
    c = 0
    core = {}
    first = True
    
    while len(vertices) > 0:
        
        # compute a dict with the degrees of nodes in vertices
        if first: # we build the dictionnary of degrees only one time then we will reduce it step by step
            deg = {}
            for e in vertices:
                deg[e] = len(G[e])
                for v in G[e]:
                    deg.setdefault(v,0)
                    deg[v]+=1
            first = False
            degree = deg
            
        # find a vertex with minimum degree
        vertices.sort(key = lambda x: deg[x])
        v = vertices[0]
        print(v)
            
        c = max(c,deg[v])
        
        # updating deg
        del deg[v]
        for neighbour in edges[v]:
            deg[neighbour]-=1
        
        #updating edges
        del edges[v]
        vertices.sort()
        for s in vertices[:vertices.index(v)]:
            tmp = edges[s]
            try:
                del (edges[s])[tmp.index(v)]
            except:
                pass
            
        #updating vertices
        del vertices[vertices.index(v)]
        
        core[v] = i
        i = i-1
        print(i)
    
    return(core, c, degree) # Returning the core values list for every vertex and the core value of the graph, and the list of degrees


# In[4]:


import math
from math import factorial

def density(G, nb_edges): # simple function to return the average degree density and the edge density of a graph
    number_of_nodes = len(list(G.keys()))
    max_number_of_edges = math.factorial(nb_edges)
    
    avg_deg_density = nb_edges/number_of_nodes
    edge_density = nb_edges/max_number_of_edges
    
    # Rk : for a real world graph, the edge density is going to be 0, since a lot of links do not exist. The edge_density
    # is useful for a subgraph when you want to see if it is denser than another subgraph for example, which is our problem
    return (avg_deg_density, edge_density)


# In[45]:


core, c, degree = core_decomposition(G) # Do not try this, it will take several hours with the amazon graph because of Python's pace...
# I tried but I stopped after getting 15,000 nodes deleted from the initial list out of 287000 in 1h30


# In[58]:


# Now we want to see the the coreness of each node as a function of the degree and see if there are outliers
import pandas as pd
import matplotlib.pyplot as plt

ID = pd.read_csv("ID.txt", sep = "\t", names = ['names'], low_memory=False)

ID['names'] = ID['names'].apply(lambda x: x.split(" ",1)[1]) #now each line corresponds to a node and its name

plt.plot(degree,core) # in order to check the outliers

