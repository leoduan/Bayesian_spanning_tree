```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
```


```python
p =200
```


```python
import sklearn

np.random.seed(9)

# mu = sklearn.datasets.make_blobs(n_samples=p, n_features=2, centers=3,shuffle=False)
mu = sklearn.datasets.make_moons(n_samples=p, noise=0.05,shuffle=False)

Y = mu[0]

plt.scatter(Y[:,0],Y[:,1])
```




    <matplotlib.collections.PathCollection at 0x7f1314441a90>




    
![png](run_example_files/run_example_2_1.png)
    



```python
import numba
from numba import jit

import importlib
import gibbs_sampling as gs
```


```python
# importlib.reload(gs)
```


```python
tree = gs.SpanningTree(Y)
```


```python
# burn-in
# adjust step_size to have the acceptance rate close to 0.3

_ = tree.runMCMC(100, step_size = 0.02)
```

    99
    0.4



```python
# collect Markov chain samples

trace = tree.runMCMC(1000, step_size = 0.02)
```

    99
    199
    299
    399
    499
    599
    699
    799
    899
    999
    0.331



```python

```


```python
trace_A = [ gs.getA(trace[i][0]) for i in range(1000)]
```


```python
trace_tau = np.array([ trace[i][3] for i in range(1000)])
```


```python
mean_tau = np.mean(trace_tau)
```


```python
# use the posterior mean of tau to quickly estimate the marginal connecting probability
tree.params[3] = mean_tau
prob = tree.computeMarginalProb()
```


```python
degree = np.vstack([trace_A[i].sum(0) for i in range(1000)])
```


```python

```


```python
from statsmodels.tsa.stattools import acf
import arviz
```


```python
acf_mat = np.vstack([ acf(degree[:,i], fft=False, nlags=40) for i in range(p)])

acf_mat[np.isnan(acf_mat)]=0

acf_mat[:,0]=1

ess = np.stack([arviz.ess(degree[:,i]) for i in range(p)])
```


```python

```


```python
fig, ax = plt.subplots(1,3, gridspec_kw={'width_ratios': [1, 1,1] })
fig.set_size_inches([9,3])

ax[0].plot(degree[:,2])
ax[0].set_title("Traceplot", y=-0.3)
# ax[0].plot(np.arange(2000),degreee[:,299]+1)

ax[1].boxplot(acf_mat[:,:40],  showfliers=False, )
ax[1].plot(np.arange(40)+1, acf_mat[:,:40].mean(0), color='red')
ax[1].set_xticks( np.arange(8)*5+1)
ax[1].set_xticklabels(np.arange(8)*5)
ax[1].set_title("ACF", y=-0.3)

ax[2].boxplot(ess/1000,  showfliers=False, )
ax[2].set_xticks( [1])
ax[2].set_xticklabels([""])
ax[2].set_title("ESS per iteration", y=-0.3)
fig.tight_layout(pad=1)
# fig.savefig("benchmark_gibbs.png")
```


    
![png](run_example_files/run_example_18_0.png)
    



```python
A1 = gs.getA(trace[0][0])
A2 = gs.getA(trace[499][0])
A3 = gs.getA(trace[999][0])
```


```python
# from pylab import rcParams
# rcParams['figure.figsize'] = 10, 8
# rcParams['figure.dpi'] = 300
```


```python
import networkx as nx
```


```python
width =2
```


```python
from matplotlib import cm
cmap = cm.get_cmap('Reds', 10)

```


```python
def pltGraph(A, color='r', usingWeight=True):
    M= nx.Graph(A)
    edges = M.edges()
    weights = [ np.log(prob[u][v]) for u,v in edges]
    
    if usingWeight:
        nx.draw(M,pos=Y,edge_color=weights,width=4, edge_cmap=cmap, node_size=100)
    else:
        nx.draw(M,pos=Y,edge_color=color,width=4, node_size=100)
```


```python
f = plt.figure()
pltGraph(A1)
# f.savefig("moons1.png")
```


    
![png](run_example_files/run_example_25_0.png)
    



```python
f = plt.figure()
pltGraph(A3,'red')
# f.savefig("moons2.png")
```


    
![png](run_example_files/run_example_26_0.png)
    



```python
import seaborn as sns
```


```python
cmap = cm.get_cmap('Blues', 20)
```


```python
prob[prob<1E-3]=np.nan
```


```python
f = plt.figure(figsize=[4,3])

plt.imshow(prob, vmin=0.0,vmax=1,cmap=cmap)
plt.colorbar()

# f.savefig("moonsMarginal.png")
```




    <matplotlib.colorbar.Colorbar at 0x7f12fb2f06d0>




    
![png](run_example_files/run_example_30_1.png)
    

