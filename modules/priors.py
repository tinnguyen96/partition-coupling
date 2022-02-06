## Todo:
    ## add sanity check for init_swap function.
from sklearn.cluster import KMeans
import numpy as np
import igraph

import partitions
# import sampler_single

# graph coloring

def greedyColoring(g, nColors):
    """
    Greedy coloring of g using nColors using the 0,1,2,... ordering of vertices. 
    If coloring is possible, return coloring. 

    Return
        possible: boolean
        labelList: list of LabelHelper objects
        clusts: (nColors,) list of sets
    """
    nvertices = g.vcount()
    vertexColors = -np.ones([nvertices], dtype=int)
    color_ids = set() 
    for ivertex in range(nvertices):
        n_i = igraph.Graph.neighbors(g, ivertex)
        legal_colors = color_ids.difference(vertexColors[n_i])
        if len(legal_colors) == 0:
            new_color_id = len(color_ids)
            color_ids.add(new_color_id)
            legal_colors.add(new_color_id)
        vertexColors[ivertex] = min(legal_colors)

    nColorsUsed = len(set(vertexColors))
    if (nColorsUsed <= nColors):
        possible = True
        labelList = [0 for _ in range(nvertices)]
        for label in range(nColorsUsed):
            labelObj = partitions.LabelHelper(label)
            equalIndices = np.argwhere(vertexColors == label)[:,0]
            for idx in equalIndices:
                labelList[idx] = labelObj
        _ , clusts = partitions.labelHelpersToLabelsAndClusts(labelList)
    else:
        possible = False
        labelList, clusts = None, None
    return possible, labelList, clusts

def SixNodeDegFourRegularinit(initType):
    ncolors = 4
    if (initType == "2with4"):
        init_state = np.array([3, 1, 0, 2, 0, 1])
    else:
        assert initType == "2separate4"
        init_state = np.array([0, 3, 2, 0, 1, 3])
    return init_state, ncolors

# def allDiffInit(g):
#     nvertices = g.vcount()
#     ncolors = nvertices
#     vertexColors = np.arange(0, nvertices)
#     return vertexColors, ncolors

def paired_init(g, joint_init, rng):
    base, ncolors = rinit(g)
    if (joint_init == "notebook"):
        Y = [base.copy()]
        nmcmc = 1000
        colors_history = [base.copy()]
        for _ in range(nmcmc):
            vertexColors_new = single_kernel(g, ncolors, colors_history[-1].copy(), None, rng)
            colors_history.append(vertexColors_new)
        X = [base.copy(), colors_history[-1]]
    else:
        assert joint_init == "lag=1" 
        Y = [base.copy()]
        X = [base.copy(), gibbs_sweep_single(g, ncolors, base.copy(), rng)] # X is one step ahead of Y
    return X, Y, ncolors

# mixture model

## initial distributions for marginalized out means
def kmeans_init(data, alpha, k_size):
    """    
    Inputs:
        data: (N, d) array, observations
        alpha: scalar, concentration of DP
        k_size: scalar, number of clusters for kmeans
        
    Output:
        k-means clustering of data using k_size clusters.
    """
    N = data.shape[0]
    n_clusters = k_size
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    z0 = kmeans.labels_
    return z0

def crp_prior(N, alpha, rng):
    """
    Input:
        N: number of observations
        alpha: scalar, concentration of DP

    Output:
        a sample from CRP prior
        
    There's probably a simpler way to do this.
    """
    z0 = np.array([0])
    for n in range(1,N):
        clusts = partitions.z_to_clusts(z0)
        clusts_as_lists = [list(clust) for clust in clusts]
        clusts_sizes_with_labels = [(len(clust),z0[clust[0]]) for clust in clusts_as_lists]
        probs_with_label = [(clust[0]/(n+alpha), clust[1]) for clust in clusts_sizes_with_labels] # adding to old clusters
        probs_with_label.append((alpha/(n+alpha),len(clusts))) # new cluster
        probs = [x[0] for x in probs_with_label]
        new_point = probs_with_label[rng.choice(len(probs), p=probs)][1]
        z0 = np.append(z0,new_point)
    return z0

def initialPartitionAndMeans(data, initZ, alpha, initType, rng):
    """initialPartitionAndMeans() initializes the Markov chain over DPMM 
    clustering models.

    Args:
        data: (N,D) array, observations
        initZ: (N,) array 
            if None, use initType to set initial state
        alpha: DPMM concentration
        initType: str, how to generate first sample
        rng: bitGenerator

    Output:
        labelList: (N,) list, each is a labelHelper object
        clusts: (K,) list of sets
        clustSums: (K,) list of (D,) array
        means: (K,) list of (D,) array
    """
    Ndata = data.shape[0]

    if initZ is None:
        if initType == "allSame":
            z0 = np.zeros(Ndata, dtype=np.int)
        elif initType == "crpPrior":
            z0 = crp_prior(Ndata, alpha, rng)
        else:
            typ, k_size = initType.split("=")
            assert typ == "kmeans"
            z0 = kmeans_init(data, alpha, int(k_size))
    else:
        z0 = initZ.copy()

    nLabels = len(np.unique(z0))
    labelList = [0 for _ in range(Ndata)]
    clustSums, means = [0 for _ in range(nLabels)], [0 for _ in range(nLabels)]
    for label in range(nLabels):
        labelObj = partitions.LabelHelper(label)
        equalIndices = np.argwhere(z0 == label)[:,0]
        for idx in equalIndices:
            labelList[idx] = labelObj
        clustSums[label] = np.sum(data[equalIndices,:], axis=0)
        means[label] = np.mean(data[equalIndices,:], axis=0)
    _, clusts = partitions.labelHelpersToLabelsAndClusts(labelList)

    return labelList, clusts, clustSums, means

# def init_swap(single_transition, pi0):    
#     """
#     Jointly initialize the X and Y chains to induce anti-correlated correction terms.
    
#     Args:
#         single_transition: lambda function, marginal Gibbs sweep, typically gibbs_sweep_single()
#             from sampler_single.py
#         pi0: lambda function, initial distribution --- each call generates a sample
#     """
    
#     a0, b0 = pi0(), pi0()
    
#     a1 = single_transition(a0)
#     b1 = single_transition(b0)
#     b2 = single_transition(b1)
    
#     X1,X2 = [a0,a1], [b0]
#     Y1,Y2 = [b1,b2], [a1]
    
#     return X1,X2,Y1,Y2

## ----------------------------------------------------------------------
## track means
# def init_centers(z, D, sd0, rng):
#     """
#     Draw cluster means from the prior distribution.
#     """
#     nClusts = max(z)+1
#     mu0 = np.zeros(D)
#     centers = [rng.multivariate_normal(mu0,cov=sd0**2*np.eye(D)) for i in range(nClusts)]
#     assert len(centers) == nClusts
#     assert centers[0].shape[0] == D
#     return centers 

# def pi0_withMeans(data, sd, sd0, alpha, pi0_its, initType, rng):
#     """
#     Draw the partitions and the cluster means separately.
#     """
#     z0 = pi0(data, sd, sd0, alpha, pi0_its, initType, rng)
#     centers = init_centers(z0, data.shape[1], sd0, rng)
#     return z0, centers