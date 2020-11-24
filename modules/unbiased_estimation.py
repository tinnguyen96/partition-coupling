import numpy as np
import utils
import time
import time
try:
    clock = time.clock
except AttributeError:
    clock = lambda : time.clock_gettime(1)


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

def run_two_chains(m, pi0, single_transition, double_transition, time_budget):
    """run_two_chains runs two coupled chains for at least m steps and until
    they meet in the space of partitions, or if time_budget has run out

    Args:
        m: minimum number of iterations
        pi0: initial distribution
        single_transition: marginal Gibbs sweep
        double_transition: coupled Gibbs sweep
        time_budget: scalar, amount of processor time to attempt coupling

    Returns:
        States of both Markov chains, meeting time (None if meeting
        didn't happen before time_budget) and processor time after
        each sweep.
    """
    time_elapsed_list = []
    st = clock()
    # Sample X_0 and Y_0
    X, Y  = [pi0()], [pi0()]

    # Advance X by one step
    X.append(single_transition(X[-1]))
    dists = [utils.dist_from_labeling(X[-1], Y[-1])]

    # Run for m iterations / until coupled
    t = 1
    while dists[-1] != 0 or t<m:
        if dists[-1]!=0:
            # until chains have met, run double transition
            Xt, Yt = double_transition(X[-1], Y[-1])
        else:
            # once chains have met, we only only need marginal due to
            # faithfulness.
            Xt = single_transition(X[-1])
            Yt = Xt.copy()
        X.append(Xt)
        Y.append(Yt)
        dist = utils.dist_from_labeling(X[-1], Y[-1])
        dists.append(dist)
        t += 1
        time_elapsed = clock()-st
        time_elapsed_list.append(time_elapsed)
        if (time_elapsed >= time_budget):
            break

    # used up compute time without meeting or if we haven't evolved enough
    # sweeps
    if (dists[-1] != 0 or t < m):
        tau = None
    else:
        dists = np.array(dists)
        tau = np.where(dists==0)[0][0]
    return X, Y, tau, time_elapsed_list

def unbiased_est(k, h, m, X, Y, tau):
    """computes an unbiased estimate from two coupled chains following
    Jacob 2020 equation 2.1

    """
    if tau is None:
        return None
    else:
        # check that the first chain was run an extra iteration
        assert len(X) == len(Y)+1

        # check that burn in is less than number of MC iterations
        assert m > k
        assert len(X) >= m

        # compute first term (usual MCMC estimate)
        term1 = np.mean([h(x) for x in X[k:m+1]], axis=0) # X_k to X_m (remember that X[0] is X_0)

        # compute second term (bias correction)
        ls = np.arange(k+1, tau)
        term2_scalings = np.array([min([1, (l-k)/(m-k+1)]) for l in ls])
        term2_diffs = np.array([h(X[l]) - h(Y[l-1]) for l in ls])
        term2 = np.tensordot(term2_scalings, term2_diffs, axes=[[0],[0]])

        return term1 + term2
