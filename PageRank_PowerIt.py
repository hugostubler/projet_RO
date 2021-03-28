import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix, csc_matrix
import matplotlib.pyplot as plt
from numpy import linalg as LA

links = pd.read_csv('wiki_dir_links.txt', header = None, skiprows=5, sep='\t')
links = links.to_numpy()


def get_d_out(links):
  count = {}
  for i in links[:,0]:
    if i not in count:
      count[i] = 0
    count[i] += 1
  return count


def power_it(G, alpha, t):
  d_out = get_d_out(G)
  n = G.shape[0]
  P_0 = np.ones(n)*(1/n)
  for value in d_out:
    P_0[value] = P_0[value]*(1/d_out[value])
  for compteur in range(t):
    P_next = np.zeros(n)
    for i in range(G.shape[0]):
      P_next[G[i,1]] += P_0[G[i,0]]*(1/d_out[G[i,0]])
    P_next *= (1-alpha)
    P_next += (alpha/n)
    v = LA.norm(P_next, 1)
    P_next += ((1-v)/n)
    P_0 = P_next
    print(P_0.argsort()[-5:][::-1], P_0.argsort()[:5])
  return P_0



def get_in_degree(links):
  in_degree = {}
  for i in links[:,1]:
    if i not in in_degree:
      in_degree[i] = 0
    in_degree[i] += 1
  return in_degree

p_it = power_it(links, 0.15, 20)
# np.save("p_it", p_it)
p_it = np.load('p_it.npy')
descending_idx = np.argsort(p_it)[::-1]
in_degree = get_in_degree(links)
sorted_in_degree = {k: v for k, v in sorted(in_degree.items(), key=lambda item: item[1])}
corres_pr = [p_it[idx] for idx in sorted_in_degree]


plt.figure()
plt.xlabel('PageRank')
plt.ylabel('in-degree')
plt.scatter(corres_pr, sorted_in_degree.values())
plt.yscale('log')
plt.title('x = PageRank with α = 0.15, y = in-degree')
plt.savefig('PR_idegreesLogScale.png')



out_degree = get_d_out(links)
sorted_out_degree = {k: v for k, v in sorted(out_degree.items(), key=lambda item: item[1])}
corres_pr_out = [p_it[idx] for idx in sorted_out_degree]

plt.figure()
plt.xlabel('PageRank')
plt.ylabel('out-degree')
plt.scatter(corres_pr_out, sorted_out_degree.values())
plt.yscale('log')
plt.title('x = PageRank with α = 0.15, y = out-degree')
plt.savefig('PR_outdegreesLogScale.png')


p_it_alpha15 = p_it
p_it_alpha10 = power_it(links, 0.1, 6)

plt.figure()
plt.xlabel('PageRank α = 0.15 ')
plt.ylabel('PageRank α = 0.10')
plt.scatter(p_it_alpha15, p_it_alpha10)
plt.yscale('log')
plt.title('x = PageRank with α = 0.15, y = PageRank with α = 0.10,')
plt.savefig('15vs10LogScale.png')



p_it_alpha20 = power_it(links, 0.2, 6)

plt.figure()
plt.xlabel('PageRank α = 0.15 ')
plt.ylabel('PageRank α = 0.20')
plt.scatter(p_it_alpha15, p_it_alpha20)
plt.yscale('log')
plt.title('x = PageRank with α = 0.15, y = PageRank with α = 0.20,')
plt.savefig('15vs20LogScale.png')


p_it_alpha50 = power_it(links, 0.5, 6)

plt.figure()
plt.xlabel('PageRank α = 0.15 ')
plt.ylabel('PageRank α = 0.50')
plt.scatter(p_it_alpha15, p_it_alpha50)
plt.title('x = PageRank with α = 0.15, y = PageRank with α = 0.50,')
plt.savefig('15vs50.png')



p_it_alpha90 = power_it(links, 0.9, 6)

plt.figure()
plt.xlabel('PageRank α = 0.15 ')
plt.ylabel('PageRank α = 0.90')
plt.scatter(p_it_alpha15, p_it_alpha90)
plt.yscale('log')
plt.title('x = PageRank with α = 0.15, y = PageRank with α = 0.90,')
plt.savefig('15vs90LogScale.png')
