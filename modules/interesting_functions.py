## Todos:
## add sanity checks for prop_in_k_clusters and CC functions

import numpy as np
from scipy import stats

import partitions  

def LCP(clusts, k=10):
    """
    Inputs:
        clusts: (K,) list of sets

    Output:
        props: (k,) array of label assignments
    """
    # compute cluster sizes in decreasing order.
    clust_sizes = list(sorted([len(clust) for clust in clusts], reverse=True))

    nData = np.sum(clust_sizes)

    # compute proportion of datapoints assigned to up to top k clusters.
    props = np.ones(shape=[k])
    props[:len(clust_sizes)] = np.cumsum(clust_sizes)[:k]/nData
    return props

def CC(labelList, indices):
    """
    Input:
        z: (N,) vector of labels
        indices: (M,) indices of data to check co-cluster.
    Output:
        mat: (M^2,) flattened matrix whether two observations are in the same cluster
    Remark:
        we avoid reporting and saving the whole (N,N) co-clustering matrix for memory reasons 
    """
    z, clusts = partitions.labelHelpersToLabelsAndClusts(labelList)
    mat = partitions.adj_matrix(z, clusts)
    sub_mat = mat[np.ix_(indices, indices)] # (M,M)
    res = sub_mat.flatten() # (M^2,)
    return res

def posterior_predictive_density(data, z, data_new, sd, sd0, alpha):
    """posterior_predictive_density computes the posterior predictive density, conditioned
    on z and data at data_new (for the DP-mixture model).

    Args:
        data: observations
        z: assignments
        data_new: location of single observation at which to compute posterior predictive
        sd, sd0: standard deviation of noise and prior on means
        alpha: DP paramter

    Returns:
        posterior predictive density at data_new
    """
    Ndata = len(z)
    Ndata_new, D = data_new.shape

    # Initialize Posterior Predictive Density (ppd) as zeros
    ppd = np.zeros(Ndata_new)

    # Add density corresponding to new cluster
    prob_new_cluster = alpha / (Ndata+alpha)
    ppd += prob_new_cluster * stats.multivariate_normal(
        mean=np.zeros(D), cov=(sd**2 + sd0**2)*np.eye(D)).pdf(data_new)

    for clust in set(np.unique(z)):
        # pull out data in cluster 'clust'
        data_c = data[np.where(z==clust)[0]]

        # probability that new datapoint is in clust
        prob_clust = len(data_c)/(Ndata + alpha)

        # posterior over mean of clust
        post_prec = (1./(sd0**2)) + len(data_c)*(1./(sd**2))
        post_var = 1./post_prec
        post_mean = post_var*(len(data_c)*(1./(sd**2))*np.mean(data_c,axis=0))

        ppd += prob_clust * stats.multivariate_normal(
            mean=post_mean, cov=(post_var + sd**2)*np.eye(D)).pdf(data_new)
    return ppd

def posterior_predictive_density_2Dgrid(data, z, sd, sd0, alpha, n_grid_spaces=20):
    """posterior_predictive_density_grid evaluate the posterior predictive density at a grid of points.

    Args:
        (Same as posterior_predictive_density)
        n_grid_spaces: granularity of locations at which to compute density
    """
    scale = 2*sd**2 + 2*sd0**2
    delta = scale/10

    # Create vector of locations at which to query predictive distribution
    x, y = np.arange(-scale, scale, delta), np.arange(-scale, scale, delta)
    X, Y = np.meshgrid(x, y)
    X_long, Y_long = X.reshape([-1]), Y.reshape([-1])
    data_new = np.array(list(zip(X_long, Y_long)))

    # Compute predictive density and reshape into a grid for easy visualization
    ppd = posterior_predictive_density(data, z, data_new, sd, sd0, alpha)
    ppd_grid = ppd.reshape(X.shape)
    X_Y_ppd = np.array([X, Y, ppd_grid])
    return X_Y_ppd

def posterior_predictive_density_1Dgrid(grid, data, z, sd, sd0, alpha, n_grid_spaces=20):
    """posterior_predictive_density_grid evaluate the posterior predictive density at a grid of points.
    Args:
        (Same as posterior_predictive_density)
        n_grid_spaces: granularity of locations at which to compute density
    """
    # Compute predictive density and reshape into a grid for easy visualization
    ppd = posterior_predictive_density(data, z, grid[:,np.newaxis], sd, sd0, alpha)
    X_ppd = np.array([grid, ppd])
    return X_ppd