"""
Implement some functions of interest (top K proportion, predictive density)
and the unbiased estimator.
"""

# Standard libaries
import argparse
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2)
from scipy import stats
import imp
from time import time
import time
import pickle

# our implementation
import utils
from sampling import gibbs_sweep_single, gibbs_sweep_couple
from unbiased_estimation import run_two_chains, unbiased_est

## ----------------------------------------------------------------------
## functions of interest

def prop_in_k_clusters(z, k=10):
    # compute cluster sizes in decreasing order.
    clusts = utils.z_to_clusts(z)
    clust_sizes = list(sorted([len(clust) for clust in clusts], reverse=True))

    # compute proportion of datapoints assigned to up to top k clusters.
    props = np.ones(shape=[k])
    props[:len(clust_sizes)] = np.cumsum(clust_sizes)[:k]/len(z)
    return props

def co_cluster(z):
    """
    Input:
        z: (N,) vector of labels
    Output:
        mat: (N,N) matrix of whether two observations are in the same cluster
    """
    clusts = utils.z_to_clusts(z)
    mat = utils.adj_matrix(z, clusts)
    return mat

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

## ----------------------------------------------------------------------
## initial distributions

def crp_prior(N,alpha):
    """
    There's got to be a simpler way to do this.

    Input:
        N: number of observations
        alpha: scalar,

    Output:
        a sample from CRP prior
    """
    z0 = np.array([0])
    for n in range(1,N):
        # print(n)
        clusts = utils.z_to_clusts(z0)
        # print(clusts)
        clusts_as_lists = [list(clust) for clust in clusts]
        # print(clusts_as_lists)
        clusts_sizes_with_labels = [(len(clust),z0[clust[0]]) for clust in clusts_as_lists]
        probs_with_label = [(clust[0]/(n+alpha), clust[1]) for clust in clusts_sizes_with_labels] # adding to old clusters
        probs_with_label.append((alpha/(n+alpha),len(clusts))) # new cluster
        probs = [x[0] for x in probs_with_label]
        # print(probs)
        new_point = probs_with_label[np.random.choice(len(probs), p=probs)][1]
        z0 = np.append(z0,new_point)
        # print(z0)
    return z0

def pi0(data, sd, sd0, alpha, pi0_its=0, init_type="all_same"):
    """pi0 is the initial distribution for the Markov chain.

    Args:
        data: ndarray of observations
        sd, sd0, alpha: parameters of model.
        pi0_its: number of steps to take for initial distribution.

    """
    Ndata = data.shape[0]
    if init_type == "all_same":
        z0 = np.zeros(Ndata, dtype=np.int)
    else:
        assert init_type == "crp_prior"
        z0 = crp_prior(Ndata, alpha)

    for i in range(pi0_its): z0 = gibbs_sweep_single(data, z0.copy(), sd, sd0, alpha)
    return z0

## ----------------------------------------------------------------------
## estimators

def unbiased_est_crp(k, h, m, data, sd, sd0, alpha, time_budget, pi0_its=0,
        init_type="all_same", coupling="Optimal"):#
    """unbiased_est produces an unbiased estimate of a functional using the approach of Jacob 2020

    Args:
        k: # burn-in iterations
        h: lambda function of interest (of labelings z)
        m: minimum iterations
        data: observations
        time_budget: scalar, how much processor time we can afford to constructor estimator
        pi0_its: number of iterations to run before coupling
        coupling: either maximal or optimal
        init_type: whether to initialize all labels as the same, or from the
            CRP prior.

    Returns: unbiased estimate and number of steps before meeting. If meeting had not happened
        by time_budget, we just return Nones.

    Remarks:
        there is the risk that even one estimator will take a long time to compute, since
        the meeting time can be large. So we need to cap the runtime explicitly.
    """
    # Define initial distribution
    pi0_dp = lambda : pi0(data, sd, sd0, alpha, pi0_its, init_type)

    # Define marginal and coupled transitions
    gibbs_sweep = lambda z: gibbs_sweep_single(data, z.copy(), sd, sd0, alpha)
    coupled_gibbs_sweep = lambda z1, z2: gibbs_sweep_couple(
        data, z1.copy(), z2.copy(), sd, sd0, alpha, coupling=coupling)

    # Run coupled chains
    X, Y, tau, elapsed_time_list = run_two_chains(m, pi0_dp, gibbs_sweep, coupled_gibbs_sweep, time_budget)

    # Compute unbiased estimate
    H_km = unbiased_est(k, h, m, X, Y, tau)

    return H_km, X, Y, tau, elapsed_time_list

# Usual MCMC estimate for comparison. Should we do thinning of the single ergodic average?
def usual_MCMC_est_crp(k, h, m, data, sd, sd0, alpha, init_type="crp_prior"):
    """
    Inputs:
        k: # burn-in iterations
        h: lambda function of interest (of labelings z)
        m: scalar, number of sweeps
        data: observations
        sd, sd0, alpha: hyper-params

    Outputs:
        (X, mean): states of Gibbs sampler (as a list) and ergodic average
    """
    if init_type == "all_same":
        X = [np.zeros(data.shape[0], dtype=int)]
    else:
        assert init_type == "crp_prior"
        X = [crp_prior(data.shape[0],alpha)]
    for i in range(m):
        X.append(gibbs_sweep_single(data, X[-1].copy(), sd, sd0, alpha))
        if (i % 500 == 0):
            print("Finished %d iterations" %i)
    ests = [h(x) for x in X[k:]]
    mean = np.mean(ests, axis=0)
    return (X, mean)

# sanity checks -------

if __name__ == "__main__":
    
    data = stats.norm(loc=0, scale=5).rvs(size=100)
    z = np.ones(len(data))
    sd = 2.0
    sd0 = 1.0
    alpha = 1.0
    density_over_grid = posterior_predictive_density_1Dgrid(data[:,np.newaxis], z, sd, sd0, alpha)
    
    plt.figure()
    plt.plot(density_over_grid[0,:], density_over_grid[1,:])
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("Test of posterior predictive evaluation function")
    savefigpath = "sanity_checks/test_posterior_predictive_density_1Dgrid.png"
    print("Will save figure to %s" %savefigpath)
    plt.savefig(savefigpath)
    