"""utils.py contains utility functions used for simulating and evaluating
couplings of Gibbs samplers on partitions."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import ot

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

def adj_dists(A1, A2):
    """adj_dists computes Hamming distance between adjacency matrices.

    This is also the squared distance in Frobenius norm or matrices with {0, 1} valued entries.
    """
    return np.sum(A1!=A2)

def adj_dists_fast(clusts1, clusts2):
    """computes distance between adjacency matrices implied by partions without instantiating them.

    This is pricesly the metric in Mirkin and Chernyi (1)

    This is faster than first computing adjacency matrices and then using
    adj_dists when the number of clusters is small.

    I got here through somewhat convoluted reasoning, but I'm pretty sure
    this should work (and it seems to work empirically).

    References:
    (1) BG Mirkin and LB Chernyi. Measurement of the distance between distinct
    partitions of a finite set of objects. Autom Tel, 5:120–127, 1970.
    """
    # sum of square set sizes
    dist = sum(len(c)**2  for c in clusts1) + sum(len(c)**2 for c in clusts2) - 2*sum(
            len(c1.intersection(c2))**2 for c1 in clusts1 for c2 in clusts2)
    return dist

def pairwise_dists(clusts1, clusts2, intersect_sizes=None, allow_new_clust=True):
    """pairwise_dists computes the __increase__ in distance between the
    partitions defined by clusts1 and clusts2 upon the addition of an additional
    point to either an existing cluster or adding a new cluster.

    Notably, computing the increase is sufficient for solving the optimal
    transport problem.


    If provided the sizes of the intersections between each pair of clusters
    (one from clusts1 and another from clusts2) are used to facilite fast
    computation.  In particular, computing the sizes of intersections from
    scratch scales linearly with the cluster sizes.  By contrast, by
    dynamically updating these sizes whenever the cluster assignments are
    modified we can compute pairwise distances with scaling that does not
    depend on cluster size.

    Args:
        clusts1, clusts2: lists of sets of the ids of datapoints in each
            cluster.
        intersect_sizes: np.array of sizes of intersections of size
            len(clusts1) by len(clusts2).
        allow_new_clust: whether to additionally compute distances for
            adding points to a new cluster, beyond those represented in
            clusts1 and clusts2.

    Returns: np.array of increases in distance between two partitions when a
        new point is assigned to the corresponding clusters.  When
        allow_new_clust is True, this is of shape [len(clusts1) + 1, len(clusts2) + 1],
        or shape [len(clusts1), len(clusts2)] otherwise.
    """
    clusts1, clusts2 = list(clusts1), list(clusts2)
    if allow_new_clust:
        # Handle possibility of adding a new cluster by adding a new empty
        # cluster.  Since all copies are local, this does not impact out of
        # scope variables.
        clusts1.append(set())
        clusts2.append(set())

        # Appropriately expand intersection sizes if given
        if intersect_sizes is not None:
            intersect_sizes = pad_with_zeros(intersect_sizes, axis=0)
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
            # cardinality of the intersection.  See also Mirkin and Chernyi.
            dists[k, m] = len(clust1) + len(clust2) - 2*n_intersect

    # multiply distances by two, since adjacency matrix is symmetric.  The
    # previously computed distances correspond to differences in just the
    # row or column (not both).
    dists *= 2
    return dists

def plot_clusts(data, clusts, ax=None):
    """plot_clusts plots data colored by cluster for the Dirichlet process
    mixture model.
    """
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

def weights_n(data, clusts, sd, sd0, alpha, n):
    """weights_n copmutes probabilities of assignments for the Gibbs
    sampler.

    This code is for a single chain -- for coupled chains, this will be
    called twice (once for each chain).

    This code is adapted & translated to Python from
    github.com/tbroderick/mlss2015_bnp_tutorial/blob/master/ex5_dpmm.R,
    A tutorial by Tamara Broderick on Bayesian nonparametrics at the 2015
    machine learning summer school.

    This does the same as weights_n but takes advantage of diagonal
    structure of covariance matrices for faster computation.

    Args:
        data: numpy array of data of shape [N, D]
        clusts: list of the memberships of each cluster (list of list of
            int)
        sd, sd0: observation variance and component mean variance (scalars)
        alpha: DP parameter controlling dispersion of mixing proportions
        n: index of data-point for which to compute Gibbs conditional
    """
    # dimension of the data points
    data_dim = data.shape[1]

    # cluster-specific variance and precision
    Sig, Prec = sd**2, sd**(-2)

    # prior variance and precision
    Sig0, Prec0 = sd0**2, sd0**(-2)

    # prior mean on cluster parameters
    mu0 = np.zeros(data_dim)
    Nclust = len(clusts)  # initial number of clusters

    # unnormalized log probabilities for the clusters
    log_weights = np.zeros(Nclust + 1)
    # find the unnormalized log probabilities
    # for each existing cluster
    for c in range(Nclust):
        clust_c = list(clusts[c])
        c_Precision = Prec0 + len(clust_c) * Prec
        c_Sig = 1./c_Precision

        # find all of the points in this cluster
        # sum all the points in this cluster
        sum_data = np.sum(data[clust_c], axis=0)
        c_mean = c_Sig*(Prec *sum_data + Prec0*mu0)
        log_weights[c] = np.log(len(clust_c)) + np.sum(stats.norm.logpdf(
            x=data[n], loc=c_mean, scale=np.sqrt(c_Sig + Sig)))

    # find the unnormalized log probability
    # for the "new" cluster
    log_weights[Nclust] = np.log(alpha) + np.sum(stats.norm.logpdf(
        x=data[n], loc=mu0, scale=np.sqrt(Sig0 + Sig)))

    # transform unnormalized log probabilities
    # into probabilities
    max_weight = max(log_weights)
    log_weights = log_weights - max_weight
    loc_probs = np.exp(log_weights)
    loc_probs = loc_probs / sum(loc_probs)
    return loc_probs

def z_to_clusts(z, total_clusts=None):
    """z_to_clusts computes the datapoints in each cluster

    Args:
        z: array of labelings of the n datapoints.  Indexing starts at zero.
        total_clusts: if not None, fill up to that number with empty
            clusters.

    Returns:
        list of cluster memberships, e.g. second element is a list of
            indices of datapoints in the second cluster.
    """
    clusts = []
    N_clust = max(z)+1
    for clust in range(N_clust):
        clusts.append(set(np.where(z==clust)[0]))

    # make sure we've accounted for all datapoints
    assert sum(len(clust_k) for clust_k in clusts) == len(z)

    if total_clusts is not None and len(clusts) < total_clusts:
        clusts.extend([set() for _ in range(total_clusts - len(clusts))])
    return clusts

def dist_from_labeling(v1, v2):
    """dist_from_labeling computes the distance between partitions implied by the labeling

    cluster memberships implied by labelings are first computed, and then
    used to calculate a partition based distance metric.

    Args:
        v1, v2: label based representation partitions.

    Returns:
        distance between implied partitions.
    """
    clusts1, clusts2 = z_to_clusts(v1), z_to_clusts(v2)
    dist = adj_dists_fast(clusts1, clusts2)
    return dist

### Partition couplings
def naive_coupling(loc_probs1, loc_probs2):
    """naive_coupling computes a common random number generator coupling of
    the two provided discrete marginals.

    Args:
        loc_probs1, loc_probs2: marginal PMFs
    Returns: sample from the coupling.
    """
    # use same random number at each step
    u_In = np.random.uniform()
    newz1 = np.where(u_In<np.cumsum(loc_probs1))[0][0]
    newz2 = np.where(u_In<np.cumsum(loc_probs2))[0][0]
    return newz1, newz2

def max_coupling(loc_probs1, loc_probs2):
    """max_coupling samples from a maximum coupling of two marginals.

    If one has larger support than the other, the one with smaller support is
    interpreted to place zero mass on the final atoms of the other
    distribution.

    Args:
        loc_probs1, loc_probs2: marginal PMFs
    Returns: sample from the coupling.
    """
    # compute overlap pmf
    min_clusters = min([len(loc_probs1), len(loc_probs2)])
    overlap = np.min([loc_probs1[:min_clusters], loc_probs2[:min_clusters]], axis=0)
    overlap_size = np.sum(overlap)
    overlap_size = np.min([1.0, overlap_size]) # protect from rounding error
    if np.random.choice(2, p=[1-overlap_size, overlap_size]) == 1:
        newz = np.random.choice(min_clusters, p=overlap/overlap_size)
        return newz, newz

    # sample from complements independently
    loc_probs1[:min_clusters] -= overlap
    loc_probs1 /= (1-overlap_size)

    loc_probs2[:min_clusters] -= overlap
    loc_probs2 /= (1-overlap_size)
    newz1 = np.random.choice(len(loc_probs1), p=loc_probs1)
    newz2 = np.random.choice(len(loc_probs2), p=loc_probs2)
    return newz1, newz2

def optimal_coupling(probs1, probs2, pairwise_dists, normalize=True, seed=None, change_size=100):
    """optimal_coupling samples from a coupling that (approximately) minimizes the
    average distance between variables. report the sample and the resulting distance.

    Args:
      probs1: np.array (K,), loc_probs1[i] = prob mass at some x[i]

      probs2: np.array (M,), loc_probs2[j] = prob mass at some y[j]

      pairwise_dists: np.array (K,M), distances (up to an additive constant) beween
          d(x[i], y[j]). User should check that all distances are positive.

      normalize: boolean, whether to normalize pairwise_dists by the largest value,
          which is suggested to avoid numerical issues

      change_size: scalar, if max(K,M) < change_size then sample from exact
          optimal coupling, otherwise solve a Sinkhorn regularized optimal transport
          problem, which is faster but doesn't give the best coupling.

    Returns: sample from the coupling
    """
    K = probs1.shape[0]
    M = probs2.shape[0]
    prob_size = max([K,M])

    if normalize:
        our_dists = pairwise_dists/pairwise_dists.max()
    else:
        our_dists = pairwise_dists

    if (prob_size < change_size):
        coupling_mat = ot.emd(probs1, probs2, our_dists)
    else:
        print("running sinkhorn")
        tol = 0.01 # smaller tol means the coupling returned by Sinkhorn is closer to the optimal one
        coupling_mat = ot.sinkhorn(probs1, probs2, our_dists, reg=tol)

    ## flatten coupling_mat, use cumsum to sample, then unravel
    flat_mat = coupling_mat.flatten()

    if seed is not None:
        print("using constant seed")
        np.random.seed(seed)
    sample_as_1d = np.random.choice(flat_mat.shape[0], p=flat_mat)
    sample = np.unravel_index(sample_as_1d,shape=[K,M])
    if False:
        print("pairwise_dists", pairwise_dists)
        print("\n\nprobs1: %s\nprobs2: %s\nsample: %s"%(str(probs1), str(probs2), str(sample)))
    return sample

def meeting_times_plots(
    traces_by_coupling, times_by_coupling, couplings_plot=['Optimal', 'Maximal', 'Common_RNG'],
    couplings_colors=['g','b','k'], nbins=10, title=None, save_path=None,
    alpha=1.0, linewidth=0.04, n_traces_plot=10, max_iter=None,
    iter_interval=None, max_time=None, plot_iter_not_time=False):
    """meeting_times_plots generates plots used in figures 1 for the DP-MM
    and graph-coloring simulations.

    Arguments:
        traces_by_coupling: list of list of traces (distance by iteration) for each
            simulation, the outer list in coupling type.
        times_by_coupling: list of list of meeting times (in seconds), the outer list in coupling type.
        other args control the appearence of the plot
    """

    # two subplots, for half of a one column page --> width = 2.9", Height 2.0"
    f, axarr = plt.subplots(ncols=2, figsize=[2.9, 2.0], dpi=300)

    # First plot Traces
    ax = axarr[0]
    max_iter_observed = 0
    for coupling, color in list(zip(couplings_plot, couplings_colors))[::-1]:
        traces = traces_by_coupling[coupling]
        perm = np.random.permutation(len(traces))
        traces = [traces[i] for i in perm]
        for trace in traces[:n_traces_plot]:
            if max_iter is not None:
                trace = trace[:max_iter]
            if iter_interval is not None:
                trace = list(trace[::iter_interval])+[trace[-1]]
            if len(trace) > max_iter_observed:
                max_iter_observed = len(trace)
            ax.plot(trace, c=color, linewidth=linewidth, alpha=alpha)
        ax.set_xlabel("Iteration", labelpad=-1)
        ax.set_xlim([0, max_iter_observed])
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='x', pad=-0.9)
        ax.tick_params(axis='y', pad=-2, rotation=45, labelsize='small')

        ax.set_ylabel("Dist. Between Chains", labelpad=-1)

    # Second plot meeting times
    ax = axarr[1]
    if plot_iter_not_time:
        for coupling in couplings_plot:
            times_by_coupling = times_by_coupling.copy()
            times_by_coupling[coupling] = [len(trace) for trace in
                    traces_by_coupling[coupling]]

    if max_time is None:
        max_time_by_coupling = [max(times_by_coupling[coupling]) for coupling in couplings_plot]
        max_time_plot = max(max_time_by_coupling)
    else:
        max_time_plot = max_time

    bins = np.linspace(0, max_time_plot, nbins)
    ax.hist(list(np.clip(times_by_coupling[coupling],a_max=max_time_plot,
        a_min=0) for coupling in couplings_plot),
                  bins=bins, label = couplings_plot, color=couplings_colors)
    ax.tick_params(axis='x', pad=-0.9)
    ax.set_xlabel("Meeting Time (s)", labelpad=-1)
    ax.set_xlim([0,max(bins)])
    ax.set_yticks([])
    ax.set_ylabel("Count", labelpad=-0.5)

    if title is not None: plt.suptitle(title, y=1.08)
    plt.tight_layout(pad=0.2)
    if save_path is not None: plt.savefig(title.replace(" ", "_")+".png")
    plt.show()
