## Todo:
    ## change pop_idx to avoid relabelling many observations (better to use some pointers)
    ## add sanity check for many functions.

import copy
import numpy as np
from scipy import stats

class LabelHelper():
    def __init__(self, value):
        self.value = value

    def getValue(self):
        return self.value

    def setValue(self, newValue):
        self.value = newValue

# VI metric between partitions ---------------------------------------------
def entropy(clusts):
    """
    Inputs:
        clusts: list of list

    Output:
        interpret cluster proportions as the pmf of a discrete distribution
        and compute the entropy.  
    """
    sizes = []
    for clust in clusts:
        size = len(clust)
        sizes.append(size)
    sizes = np.array(sizes)
    N = np.sum(sizes)
    normed_sizes = sizes/N
    ent = stats.entropy(sizes)
    return N, ent

def VI_dist(clusts1, clusts2):
    """
    Compute the variation of information metric
    """
    N, ent1 = entropy(clusts1)       
    N2, ent2 = entropy(clusts2) 
    assert N == N2
    terms = []
    for c1 in clusts1:
        for c2 in clusts2:
            c1_prob = len(c1)/N
            c2_prob = len(c2)/N
            intersect = len(c1.intersection(c2))
            if intersect > 0:
                joint_term = intersect/N
                term = joint_term*np.log(joint_term/(c1_prob*c2_prob))
                terms.append(term)
      
    dist = ent1 + ent2 - np.sum(terms)
    return dist

def pairwise_VI_dists(clusts1, clusts2, n, 
                        allowNewClustX=True, allowNewClustY=True):
    """pairwise_VI_dists computes the distance between the partitions defined by clusts1 and clusts2
    upon the addition of an additional point to either an existing cluster or adding a new cluster under 
    the VI metric.
    
    n: int, index of data point to be added. User should check that n doesn't already
    belong to clusts1 or clusts2
    
    As currently implemented, not as efficient as pairwise_dists (Hamming)
    """
    clusts1, clusts2 = list(clusts1), list(clusts2)
    if (allowNewClustX):
        clusts1.append(set())

    if (allowNewClustY):
        clusts2.append(set())
        
    K, M = len(clusts1), len(clusts2)
    dists = np.zeros([K, M])
    for k, clust1 in enumerate(clusts1):
        for m, clust2 in enumerate(clusts2):
            clusts1[k] = clust1.union({n})
            clusts2[m] = clust2.union({n})
            dists[k, m] = VI_dist(clusts1, clusts2)
            # reset values
            clusts1[k] = clust1
            clusts2[m] = clust2
    return dists

# Hamming metric between partitions ---------------------------------------------
def pad_with_zeros(arr, axis):
    """pad_with_zeros returns a copy of arr with one row or column of zeros appended

    Args:
        arr: 2D np.array
        axis: axis on which to add pad (add a row if axis=0, or a column if axis=1)
    """
    assert len(arr.shape) == 2
    assert axis==0 or axis==1
    D1, D2 = arr.shape
    D1_new = D1 + (1 if axis==0 else 0)
    D2_new = D2 + (1 if axis==1 else 0)

    arr_pad = np.zeros(shape=[D1_new, D2_new], dtype=int)
    arr_pad[:D1,:D2] = arr
    return arr_pad

def adj_matrix(z, clusts):
    """adj_matrix computes the adjacency matrix implied by z and clusts"""
    Ndata = len(z)
    A = np.zeros([Ndata, Ndata])
    for clust in clusts:
        clust = list(clust)
        for i in clust:
            for j in clust:
                A[i,j] = 1.
    return A

# def adj_dists(A1, A2):
#     """adj_dists computes Hamming distance between adjacency matrices.

#     This is also the squared distance in Frobenius norm or matrices with {0, 1} valued entries.
#     """
#     return np.sum(A1!=A2)

def adj_dists_fast(clusts1, clusts2):
    """computes distance between adjacency matrices implied by partions without instantiating them.
    This is faster than first computing adjacency matrices and then using
    adj_dists when the number of clusters is small.
    """
    # sum of square set sizes
    dist = sum(len(c)**2  for c in clusts1) + sum(len(c)**2 for c in clusts2) - 2*sum(
            len(c1.intersection(c2))**2 for c1 in clusts1 for c2 in clusts2)
    return dist

def pairwise_dists(clusts1, clusts2, 
                    intersect_sizes=None, allowNewClustX=True, allowNewClustY=True):
    """pairwise_dists computes the __increase__ in distance between the partitions defined by clusts1 and clusts2
    upon the addition of an additional point to either an existing cluster or adding a new cluster under the 
    Hamming metric.

    Args:
        clusts1, clusts2: lists of sets of the ids of datapoints in each
            cluster.
        allowNewClustX: boolean, whether to additionally compute distances for
            adding points to a new cluster, beyond those represented in
            clusts1

        allowNewClustY: boolean, whether to additionally compute distances for
            adding points to a new cluster, beyond those represented in
            clusts1

    Returns: np.array of increases in distance between two partitions when a
        new point is assigned to the corresponding clusters.  When
        allowNewClustX and allowNewClustY are True, this is of shape [len(clusts1) + 1, len(clusts2) + 1].
        If allowNewClustX is true while allowNewClustY is False, this is of shape [len(clusts1) + 1, len(clusts2)] etc.
    """
    clusts1, clusts2 = list(clusts1), list(clusts2)

    if allowNewClustX:
        clusts1.append(set())
        # Appropriately expand intersection sizes if given
        if intersect_sizes is not None:
            intersect_sizes = pad_with_zeros(intersect_sizes, axis=0)

    if allowNewClustY:
        clusts2.append(set())
        # Appropriately expand intersection sizes if given
        if intersect_sizes is not None:
            intersect_sizes = pad_with_zeros(intersect_sizes, axis=1)

    K, M = len(clusts1), len(clusts2)
    dists = np.zeros([K, M])

    for k, clust1 in enumerate(clusts1):
        for m, clust2 in enumerate(clusts2):
            if intersect_sizes is not None:
                n_intersect = intersect_sizes[k, m]
            else:
                n_intersect = len(clust1.intersection(clust2))

            # number of different entries in the new row of the adjacency
            # matrix is equal to the size of the 'outersection' of the two
            # clusters to which the point would be added.  This size may be
            # computed as below, the sum of the cardinalties minus twice the
            # cardinality of the intersection.
            dists[k, m] = len(clust1) + len(clust2) - 2*n_intersect

    # multiply distances by two, since adjacency matrix is symmetric.  The
    # previously computed distances correspond to differences in just the
    # row or column (not both).
    dists *= 2
    return dists

# def overlap_score(clusts1, clusts2, perm_of_2):
#     min_nclust = min([len(clusts1), len(clusts2)])
#     score = 0
#     clusts2_reorder = [clusts2[i] for i in perm_of_2]
#     for c1, c2 in zip(clusts1[:min_nclust], clusts2_reorder[:min_nclust]):
#         score += len(c1.intersection(c2))
#     return score

# def match_clust(clusts1, clusts2):
#     perms = list(itertools.permutations(range(len(clusts2))))
#     overlap_scores = [overlap_score(clusts1, clusts2, perm) for perm in perms]
#     opt_perm = perms[np.argmax(overlap_scores)]

#     clusts2_match = [clusts2[i] for i in opt_perm]
#     Ndata = sum(len(clust) for clust in clusts2)

#     # compute cluster assignments as an array
#     z2 = -np.ones(Ndata, dtype=np.int) # unassigned datapoints index as -1
#     for c, clust in enumerate(clusts2_match): z2[list(clust)] = c

#     return z2, clusts2_match

def find_pairings(clusts1, clusts2, intersection_xsizes):
    """
    For each cluster in clusts1, find the cluster in clusts2 with the largest intersection.
    Prioritize clusters in clusts1 by their size.
    
    Inputs:
    
    Outputs:
        pairs: list of tuples, len(pairs) = min(len(clusts1), len(clust2))
        rem_clusts1: list, indices of remaining clusters in clusts1 that don't have pair
            will be [] if len(clusts1) < len(clusts2)
        rem_clusts2: vice-versa of rem_clusts1

    Runtime:
        ?
    """
    
    pairs = []
    
    clusts1_lens = np.array([len(c1) for c1 in clusts1])
    arg_sort1 = np.argsort(clusts1_lens)[::-1] # get cluster indices in descending order
    num_clusts1 = len(clusts1)
    
    num_clusts2 = len(clusts2)
    unpaired_2 = set(range(num_clusts2)) # at beginning, nothing has been paired up
    
    rem_clusts1 = []
    for i in range(num_clusts1):
        if (len(unpaired_2) == 0):
            rem_clusts1 = arg_sort1[i::]
            break
        else:
            ind1 = arg_sort1[i]
            intersect_in_clusts2 = intersection_xsizes[ind1,:]
            argsort_2 = np.argsort(intersect_in_clusts2)[::-1]
            # find the first unpaired cluster in 2 that has large intersection size
            for cand in argsort_2:
                if cand in unpaired_2:
                    ind2 = cand
                    break
            unpaired_2.remove(ind2)
            pair = (ind1, ind2)
            pairs.append(pair)
    
    rem_clusts2 = list(unpaired_2)
    return pairs, rem_clusts1, rem_clusts2

def zToLabelHelpers(z):
    nLabels = len(np.unique(z))
    N = len(z)
    labelList = [0 for _ in range(N)]
    for label in range(nLabels):
        labelObj = LabelHelper(label)
        equalIndices = np.argwhere(z == label)[:,0]
        for idx in equalIndices:
            labelList[idx] = labelObj
    return labelList

def zToClusts(z):
    clusts = []
    N_clust = max(z)+1
    for clust in range(N_clust):
        clusts.append(set(np.where(z==clust)[0]))

    # make sure we've accounted for all datapoints
    assert sum(len(clust_k) for clust_k in clusts) == len(z)

    return clusts 

def labelHelpersToLabelsAndClusts(labelList):
    """z_to_clusts computes the datapoints in each cluster. 
    
    Inputs:
        labelList: (N,) list, each is a LabelHelper object

    Output:
        z: (N,) array, labels
        clusts: (K,) list of sets 

    Runtime:
        O(N)

    """

    # convert to pure labels
    z = np.array([labelObject.getValue() for labelObject in labelList], dtype=np.int32)

    clusts = zToClusts(z)

    return z, clusts

# def dist_from_labeling(v1, v2):
#     """dist_from_labeling computes the distance between partitions implied by the labeling
#     """
#     clusts1, clusts2 = z_to_clusts(v1), z_to_clusts(v2)
#     dist = adj_dists_fast(clusts1, clusts2)
#     return dist

def dist_from_clusts(clusts1, clusts2):
    """
    Inputs:
        clusts1: (K,) list of sets, each is set of observations in a cluster
        clusts2: (M,) list of sets, each is set of observations in a cluster
    Runtime:
        O(KM)
    """
    dist = adj_dists_fast(clusts1, clusts2)
    return dist

def compute_tau(dists, t, m):
    """
    Inputs:
        dists: list of distances, where dists[i] = distance between X[i+1] and Y[i]
        t: scalar, number of steps either chains took
        m: scalar, minimum number of sweeps

    Output:
        tau: scalar, first time where dists = 0, measured in X's time (recall
            that X is advanced one step ahead of Y)
        hasMet: boolean, whether the chains met
    """
    if (dists[-1] != 0 or t < m):
        tau = -1; hasMet = False
    else:
        dists = np.array(dists)
        tau = np.where(dists==0)[0][0] + 1 
        hasMet = True
    return tau, hasMet

def labelOfCluster(labelList, clusts, c):
    """
    Inputs:
        labelList: (N,) list, each is a LabelHelper object
        clusts: (K,) list of set (each set is a cluster)
        c: scalar, cluster index

    Return:
        labelObj of (any) observations in cluster c.

    Runtime:
        O(1)
    """
    return labelList[next(iter(clusts[c]))]

def reassignClust(z, clusts, zi, zj):
        """
        Merges the zi cluster and the zj cluster, in the process
        changing the label vector z to not have any gaps. We choose
        the label for the new cluster to minimize the amount of work to 
        relabel. 

        Args:
            z: (N,) np array, data labels 
            clusts: (K,) list of clusters (each cluster is a set)
            zi: scalar, cluster to be removed
            zj: scalar, target of reassignment

        Output:
            nextLabelList: (N,)
            nextZ: (N,) array
            nextClusts: (K-1,) list of clusters 
        """
        nextZ = z.copy()
        nextClusts = copy.deepcopy(clusts)

        if (zi < zj):
            source = zj 
            target = zi
        else:
            source = zi
            target = zj

        # move indices from zi cluster to zj cluster
        nextZ[list(nextClusts[source])] = target
        nextClusts[target] = nextClusts[target].union(nextClusts[source])
        nextClusts[source].clear()

        # relabel the clusters
        num_clusts = len(nextClusts)
        for i in range(source, num_clusts-1):
            nextClusts[i] = nextClusts[i+1]
            nextZ[list(nextClusts[i])] = i
        nextClusts.pop() # drop last cluster

        # convert to labelList
        nextLabelList = zToLabelHelpers(nextZ)

        return nextLabelList, nextZ, nextClusts

def pop_idx(labelList, 
            clusts, means, clustSums, 
            n, obs):
    """pop_idx removes the datapoint n from the current cluster assignments.
    Make in-place changes rather than make new copies of inputs.
    Ensure that the labels from labelList are contiguous integers (0, 1, ..., K-1) even
    if popping a data point removes a whole cluster.

    Args:
        labelList: (N,) list, each is a LabelHelper object

        clusts: (K,) list of set (each set is a cluster)
        means: (K,) list of (D,) arrays, each is the cluster mean under the model (not the mean of observations)
        clustSums: (K,) list of (D, ) arrays, each is the sum of all data points in a cluster

        n: index of datapoint to remove
        obs: (D,) observation to remove

    Runtime:
        O(K + D)

    Returns: 
            (in place) updated labelList, clusters, 
            (not in place) index of removed cluster (or None if no cluster was removed)

    """
    c = labelList[n].getValue()
    
    if (n not in clusts[c]):
        z, _ = labelHelpersToLabelsAndClusts(labelList)
        print("n",n)
        print("clusts", clusts)
        print("z", z)
        raise RuntimeError
        
    labelList[n] = LabelHelper(-1.0)
    clusts[c].remove(n)
    
    if (clustSums is not None):
        clustSums[c] = clustSums[c] - obs

    # if cluster is empty, remove it and reassign others
    if len(clusts[c]) == 0:
        removed_cluster = c
        for i in range(c, len(clusts)-1):
            clusts[i] = clusts[i+1]
            labelOfCluster(labelList, clusts, i).setValue(i)
            if (clustSums is not None) and (means is not None):
                means[i] = means[i+1]
                clustSums[i] = clustSums[i+1]
        # drop last cluster
        clusts.pop(); 
        if (clustSums is not None) and (means is not None):
            means.pop(); clustSums.pop()
    else:
        removed_cluster = None
    return removed_cluster

# visualize partitions for simple cases ----------------------------------------
def plot_clusts(data, clusts, ax=None):
    if ax == None:
        ax = plt.subplot(111)
        show=True
    else:
        show = False

    for idcs in clusts:
        idcs = list(idcs)
        x_clust = data[idcs]
        ax.scatter(x_clust[:, 0], x_clust[:, 1])
    lim = np.max(np.abs(data))
    ax.set_xlim([-lim,lim])
    ax.set_ylim([-lim,lim])

    if show: plt.show()