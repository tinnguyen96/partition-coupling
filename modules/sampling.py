"""sampling.py contains methods for simulating Gibbs samplers on partitions
with and without couplings"""

import numpy as np
import utils

def pop_idx(z, clusts, n):
    """pop_idx removes the datapoint n from the current cluster assignments

    z[n] is temporarily set to max int (to cause an error if we try to access
        that cluster)

    Args:
        z: list of cluster assignments
        clusts: list clusters (each cluster is a set)
        n: index of datapoint to remove

    Returns:
        updated labels, clusters, and index of removed cluster (or None if no cluster was removed)
    """
    c = z[n]
    z[n] = np.iinfo(z.dtype).max # set
    clusts[c].remove(n)

    # if cluster is empty, remove it and reassign others
    if len(clusts[c]) == 0:
        removed_cluster = c
        for i in range(c, len(clusts)-1):
            clusts[i] = clusts[i+1]
            z[list(clusts[i])] = i
        clusts.pop() # drop last cluster
    else:
        removed_cluster = None
    return z, clusts, removed_cluster


def crp_gibbs(data, sd, sd0, initz, alpha=0.01, plot=False, log_freq=None, maxIters=100):
    """crp_gibbs runs a gibbs sampler

    The state of the chain includes z and clusts, which are redundant
    but make for easy / fast indexing.

    just setting alpha for inference.
    a small alpha encourages a small number of clusters.

    Returns:
        z  (array of cluster assignments)
    """

    # initialize the sampler
    z = np.array(initz) # don't overwrite
    clusts = utils.z_to_clusts(z)  # initial data counts at each cluster

    # set frequency at which to log state of the chain
    if log_freq is None: log_freq = int(maxIters/10)

    # run the Gibbs sampler
    for I in range(maxIters):
        # take a Gibbs step at each data point
        z = gibbs_sweep_single(data, z.copy(), sd, sd0, alpha=alpha)

        if (I%log_freq==0 or I==maxIters-1) and plot:
            clusts = utils.z_to_clusts(z)  # initial data counts at each cluster
            print("Iteration %04d/%04d"%(I, maxIters))
            utils.plot_clusts(data, clusts, ax=None)
    return z

def forward_one_chain(lag, data, sd, sd0, alpha, debug=False):
    """
    Inputs:
        lag: scalar
        data: (N,2) array
        sd, sd0, alpha: scalar
    Outputs:
    """
    Ndata = data.shape[0]
    print("Running first chain ahead by %d steps"%lag)
    initz = np.zeros(Ndata, dtype=np.int) # initialize all data points to be in one cluster
    initz1 = crp_gibbs(data, sd, sd0, initz, alpha=alpha, log_freq=None, maxIters=lag)
    initz2 = np.zeros(Ndata, dtype=np.int) # initialize all data points to be in one cluster
    if (debug):
        print("End state of 1st chain forwarding")
        print(initz1)
        print("Initial state of 2nd chain")
        print(initz2)
    return (initz1, initz2)

def gibbs_sweep_single(data, z, sd, sd0, alpha=0.01):
    """gibbs_sweep_single runs through a Gibbs sweep for a single chain for
    explort the DP-MM posterior.

    Args:
        data: np.array of observations of shape N by D, where N is the
            number of data-points.
        z: labelings at start of sweep.
        sd, sd0: likelihood and prior standard deviations.
        alpha: DP prior parameter.

    Returns:
        New labelings after Gibbs sweep.
    """
    clusts = utils.z_to_clusts(z)  # initial data counts at each cluster
    # take a Gibbs step at each data point
    for n in range(len(data)):
        # get rid of the nth data point and relabel clusters if needed
        z, clusts, _ = pop_idx(z, clusts, n)

        # sample which cluster this point should belong to
        loc_probs = utils.weights_n(data, clusts, sd, sd0, alpha, n)
        newz = np.random.choice(len(loc_probs), p=loc_probs)

        # if necessary, instantiate a new cluster
        if newz == len(clusts): clusts.append(set())

        # update cluster assignments
        clusts[newz].add(n)
        z[n] = newz
    return z

def gibbs_sweep_couple(data, z1, z2, sd, sd0, alpha=0.01, coupling="Maximal"):
    """gibbs_sweep_couple performs Gibbs updates for every datapoint,
    coupling each update across the two chains.

    Args:
        data: np.array of shape [N, D]  (N is # of data-points, D is data
            dimension)
        z1, z2: labelings of data-points, np.array of natural numbers
        sd, sd0: standard deviation of within cluster (likelihood/sd) and
            across (prior/sd0)
        alpha: DP prior parameter
        coupling: which coupling to use.  Must be "Common_RNG", "Maximal" or
            "Optimal"

    Returns:
        updated labelings for both chains

    We compute intersection sizes once at the start and then update it for
    better amortized time complexity.
    """
    # Compute clusters and intersection sizes from scratch once
    clusts1, clusts2 = utils.z_to_clusts(z1), utils.z_to_clusts(z2)  # initial data counts at each cluster
    intersection_sizes = np.array([[len(c1.intersection(c2)) for c2 in clusts2] for c1 in clusts1])

    # Take a Gibbs step at each data point
    for n in range(len(data)):
        # Get rid of the nth data point and relabel clusters if needed

        # intersection sizes with data-point n removed
        intersection_sizes[z1[n], z2[n]] -= 1

        z1, clusts1, removed_clust1 = pop_idx(z1, clusts1, n)
        z2, clusts2, removed_clust2 = pop_idx(z2, clusts2, n)

        # If a cluster was removed in either chain, remove the appropriate
        # column/row from intersection sizes
        if removed_clust1 is not None:
            intersection_sizes = np.delete(intersection_sizes, removed_clust1, axis=0)
        if removed_clust2 is not None:
            intersection_sizes = np.delete(intersection_sizes, removed_clust2, axis=1)

        # Compute marginal probabilities
        loc_probs1 = utils.weights_n(data, clusts1, sd, sd0, alpha, n)
        loc_probs2 = utils.weights_n(data, clusts2, sd, sd0, alpha, n)

        # Sample new clusters assignments from coupling
        if coupling=="Common_RNG":
            newz1, newz2 = utils.naive_coupling(loc_probs1, loc_probs2)
        elif coupling=="Maximal":
            newz1, newz2 = utils.max_coupling(loc_probs1, loc_probs2)
        else:
            assert coupling=="Optimal"
            pairwise_dists = utils.pairwise_dists(clusts1, clusts2, intersection_sizes)

            newz1, newz2 = utils.optimal_coupling(loc_probs1, loc_probs2,
                    pairwise_dists, normalize=True, change_size=100)

        # if necessary, instantiate a new cluster and pad intersection_sizes appropriately
        if newz1 == len(clusts1):
            clusts1.append(set())
            intersection_sizes = utils.pad_with_zeros(intersection_sizes, axis=0)
        if newz2 == len(clusts2):
            clusts2.append(set())
            intersection_sizes = utils.pad_with_zeros(intersection_sizes, axis=1)

        # update cluster assigments and intersection sizes
        clusts1[newz1].add(n); clusts2[newz2].add(n)
        z1[n], z2[n] = newz1, newz2
        intersection_sizes[z1[n],z2[n]] += 1

    # Check that intersection and pairwise distances have been computed and
    # tracked correctly.
    #intersection_sizes_correct = np.array( [[len(c1.intersection(c2)) for c2 in clusts2] for c1 in clusts1])
    #assert np.alltrue(intersection_sizes==intersection_sizes_correct)

    #pairwise_dists = utils.pairwise_dists(clusts1, clusts2, intersection_sizes)
    #pairwise_dists_correct = utils.pairwise_dists(clusts1, clusts2)
    #assert np.alltrue(intersection_sizes==intersection_sizes_correct)

    return z1, z2
