## Todos:
## add sanity check for coupling of beta distribution c_maximal_beta

## Known issues:
## - sinkhorn coupling runs into numerical problems where output of ot.sinkhorn
## doesm't sum up to one

import copy
import warnings
import numpy as np
import ot
import scipy

import weights

MAX_STEPS = 100

DEFAULT_C = 0.95

def indep_coupling(loc_probs1, loc_probs2, rng):
    # use independent random numbers at each step
    u1_in = rng.uniform()
    newz1 = np.where(u1_in<np.cumsum(loc_probs1))[0][0]
    u2_in = rng.uniform()
    newz2 = np.where(u2_in<np.cumsum(loc_probs2))[0][0]
    return newz1, newz2

def pure_naive_coupling(loc_probs1, loc_probs2, rng):
    # use same random number at each step
    u_In = rng.uniform()
    newz1 = np.where(u_In<np.cumsum(loc_probs1))[0][0]
    newz2 = np.where(u_In<np.cumsum(loc_probs2))[0][0]
    return newz1, newz2

def pure_maximal_coupling(loc_probs1, loc_probs2, rng):
    """
    Inputs:
        loc_probs1: np.array (K,), loc_probs1[i] = prob mass at some x[i]
        probs2: np.array (M,), loc_probs2[j] = prob mass at some y[j]

    Remark:
        the function changes the loc_probs1, so from outside should copy.
    """
    # compute overlap pmf
    min_clusters = min([len(loc_probs1), len(loc_probs2)])
    overlap = np.min([loc_probs1[:min_clusters], loc_probs2[:min_clusters]], axis=0)
    overlap_size = np.sum(overlap)
    overlap_size = np.min([1.0, overlap_size]) # protect from rounding error
    if rng.choice(2, p=[1-overlap_size, overlap_size]) == 1:
        newz = rng.choice(min_clusters, p=overlap/overlap_size)
        return newz, newz

    # sample from complements independently
    loc_probs1[:min_clusters] -= overlap
    loc_probs1 /= (1-overlap_size)

    loc_probs2[:min_clusters] -= overlap
    loc_probs2 /= (1-overlap_size)
    newz1 = rng.choice(len(loc_probs1), p=loc_probs1)
    newz2 = rng.choice(len(loc_probs2), p=loc_probs2)
    return newz1, newz2

def naive_coupling(loc_probs1, loc_probs2, nugget, rng):
    """
    Inputs:
        probs1: np.array (K,), loc_probs1[i] = prob mass at some x[i]
        probs2: np.array (M,), loc_probs2[j] = prob mass at some y[j]
        nugget: scalar, mixing amount with indep coupling
        rng: BitGenerator
    Output:
        a sample from the (1-nugget)*common_RNG + nugget*indep coupling
    """
    comps = np.asarray([nugget, 1-nugget])
    mix = rng.choice(a=2,p=comps)
    if (mix == 0):
        print("Use small indep component")
        newz1, newz2 = indep_coupling(loc_probs1, loc_probs2, rng)
    else:
        newz1, newz2 = pure_naive_coupling(loc_probs1, loc_probs2, rng)
    return newz1, newz2

def max_coupling(loc_probs1, loc_probs2, nugget, rng):
    comps = np.asarray([nugget, 1-nugget])
    mix = rng.choice(a=2, p=comps)
    if (mix == 0):
        print("Use small indep component")
        newz1, newz2 = indep_coupling(loc_probs1, loc_probs2, rng)
    else:
        newz1, newz2 = pure_maximal_coupling(loc_probs1, loc_probs2, rng)
    return newz1, newz2

def optimal_coupling(probs1, probs2, pairwise_dists,
                    normalize, reg, rng):
    """
    samples from a coupling that minimizes the average distance between variables. 
    Args:
      probs1: np.array (K,), loc_probs1[i] = prob mass at some x[i]
      probs2: np.array (M,), loc_probs2[j] = prob mass at some y[j]
      pairwise_dists: np.array (K,M), distances (up to an additive constant) beween
          d(x[i], y[j]). User should check that all distances are positive.
      normalize: boolean, whether to normalize pairwise_dists by the largest value,
          which is suggested to avoid numerical issuee
      reg: scalar, typically small value (like 1e-5), perturb the coupling_mat so that
          in theory, all transitions are possible (for coupling theory)
      rng: BitGenerator

    Returns:
      optimal coupling (U,V) has the right marginals and minimizes E[d(x[i], y[j])].
    """
    K = probs1.shape[0]
    M = probs2.shape[0]

    if normalize:
        our_dists = pairwise_dists/pairwise_dists.max()
    else:
        our_dists = pairwise_dists

    ot_sol = ot.emd(probs1, probs2, our_dists)
    indep_mat = np.outer(probs1, probs2)
    coupling_mat = (1-reg)*ot_sol + reg*indep_mat # mix with independent coupling

    ## flatten coupling_mat, use cumsum to sample, then unravel
    flat_mat = coupling_mat.flatten()

    sample_as_1d = rng.choice(flat_mat.shape[0], p=flat_mat)
    sample = np.unravel_index(sample_as_1d,shape=[K,M])

    return coupling_mat, sample, pairwise_dists[sample]

def sinkhorn_coupling(probs1, probs2, pairwise_dists,
                    normalize, reg, rng):
    """
    samples from a coupling that (approximately) minimizes the
    average distance between variables. report the sample and the resulting distance.

    Args:
      probs1: np.array (K,), loc_probs1[i] = prob mass at some x[i]
      probs2: np.array (M,), loc_probs2[j] = prob mass at some y[j]
      pairwise_dists: np.array (K,M), distances (up to an additive constant) beween
          d(x[i], y[j]). User should check that all distances are positive.
      normalize: boolean, whether to normalize pairwise_dists by the largest value,
          which is suggested to avoid numerical issue
      reg: scalar, amount of regularization in Sinkhorn
      rng: BitGenerator

    Returns:
      Sinkhorn coupling (U,V) has the right marginals and (approximately)
      minimizes E[d(x[i], y[j])].
    """
    K = probs1.shape[0]
    M = probs2.shape[0]

    if normalize:
        our_dists = pairwise_dists/pairwise_dists.max()
    else:
        our_dists = pairwise_dists

    coupling_mat = ot.sinkhorn(probs1, probs2, our_dists, reg=reg, 
                               method='sinkhorn_epsilon_scaling')
    is_close = np.allclose(np.sum(coupling_mat), 1)
    if (not is_close):
        warnings.warn("Coupling mat from Sinkhorn solver not exactly equal to 1, need to manually normalize.")
        coupling_mat = coupling_mat/np.sum(coupling_mat)
    # assert np.allclose(np.sum(coupling_mat, axis=1), probs1, rtol=1e-04, atol=1e-06)
    # assert np.allclose(np.sum(coupling_mat, axis=0), probs2, rtol=1e-04, atol=1e-06)

    ## flatten coupling_mat, use cumsum to sample, then unravel
    flat_mat = coupling_mat.flatten()

    sample_as_1d = rng.choice(flat_mat.shape[0], p=flat_mat)
    sample = np.unravel_index(sample_as_1d,shape=[K,M])

    return coupling_mat, sample, pairwise_dists[sample]

def OT_Gaussians(mean1, mean2, var1, var2, rng):
    """
    Return sample from optimal transport couping between two diagonal Gaussians
    with squared Euclidean ground metric.
    Following Remark 2.31 of Peyre and Cuturi Computational OT book. 
    
    Args:
        mean1, mean2: (D,) arrays, means of the Gaussians
        var1, var2: scalars, variances of the Gaussians
        
    Outputs:
        sample1, sample2: (D,) arrays, samples from the OT coupling
    """
    
    D = mean1.shape[0]
    # get sample from first 
    sd1 = np.sqrt(var1)
    sample1 = rng.normal(mean1, scale=sd1)
    # find the correction matrix
    sd2 = np.sqrt(var2)
    factor = sd2/sd1
    # get sample from second
    sample2 = mean2 + factor*(sample1-mean1)
    return (sample1, sample2)


def c_maximal(c, rng, rXdist, lpdfXdist, rYdist, lpdfYdist):
    """
    c-maximal coupling is a sub-optimal maximal coupling (which maximizes the probability
    that rvs exactly equal each other) but with a time-until-sample that has a variance
    independent of how close the two distributions are. 

    Reference:
        Jacob 2020, Mathieu gerber and Anthony Lee discussion.

    Inputs:
        c: scalar, number less than 1 (1 reduces to the regular maximal coupling)
        rXdist: lambda function, taking in BitGenerator to draw samples from the X distribution
        lpdfXdist: lambda function to evaluate log pdf of the X distribution
        rYdist: lambda function to draw samples from the Y distribution
        lpdfYdist: lambda function to evaluate log pdf of the Y distribution

    Output: 

    Remark:
        if we take too many rejection sampling steps, do independent coupling instead
    """
    assert c <= 1.0

    X = rXdist(rng)
    isEqual = False

    W = np.exp(min(np.log(c),lpdfYdist(X)-lpdfXdist(X)))
    U = rng.uniform()
    if (U <= W):
        Y = copy.deepcopy(X)
        isEqual = True
    else:
        it = 0
        while (True):
            Ystar = rYdist(rng)
            Wstar = rng.uniform()
            it += 1
            criterion = np.log(Wstar) - np.log(c) - lpdfXdist(Ystar) + lpdfYdist(Ystar)
            if (criterion > 0):
                Y = Ystar
                break
        if (it > MAX_STEPS):
            warnings.warn("rejection sampling in c-maximal finished, but took more than %d tries" %MAX_STEPS)
    return X, Y, isEqual


def c_maximal_beta(rng, a1, b1, a2, b2):
    """"
    Specialized version of c_maximal for gamma distributions.
    If the two gammas are very close (np.close), set c = 1.0, since c-maximal coupling
    is not faithful for c < 1.9

    Inputs:
        rng: BitGenerator
        a1, b1: scalars, a and b of first beta distribution
        a2, b2: scalars, a and b of second beta distribution
    Outputs:
    """

    beta1 = scipy.stats.beta(a = a1, b =  b1)
    lpdfXdist = lambda x: beta1.logpdf(x)
    rXdist = lambda rng: rng.beta(a = a1, b =  b1)

    beta2 = scipy.stats.beta(a = a2, b =  b2)
    lpdfYdist = lambda x: beta2.logpdf(x)
    rYdist = lambda rng: rng.beta(a = a2, b =  b2)

    if np.allclose([a1, b1], [a2, b2]):
        X = rXdist(rng)
        Y = copy.deepcopy(X)
        isEqual = True
    else:
        X, Y, isEqual = c_maximal(DEFAULT_C, rng, rXdist, lpdfXdist, rYdist, lpdfYdist)
    return X, Y, isEqual

def c_maximal_Bernoulli(rng, p1, p2):

    """
    Inputs:
        rng: BitGenerator
        a1, b1: scalars, a and b of first beta distribution
        a2, b2: scalars, a and b of second beta distribution
    Outputs:
    """

    u = rng.uniform()

    maxP = np.amax([p1, p2])
    minP = np.amin([p1, p2])

    if u <= minP:
        X, Y, isEqual = 1, 1, True
    elif u > maxP:
        X, Y, isEqual = 0, 0, True
    else: 
        X = int(u <= p1); Y = int(u <= p2)
        isEqual = False
    return X, Y, isEqual

def c_maximal_gamma(rng, shape1, rate1, shape2, rate2):
    """
    Specialized version of c_maximal for gamma distributions.
    If the two gammas are very close (np.close), set c = 1.0, since c-maximal coupling
    is not faithful for c < 1.9

    Inputs:
        rng: BitGenerator
        shape1, rate1: scalars, shape and rate of first gamma distribution
        shape2, rate2: scalars, shape and rate of second gamma distribution
    Outputs:
    """

    gamma1 = scipy.stats.gamma(a = shape1, scale =  1/rate1)
    lpdfXdist = lambda x: gamma1.logpdf(x)
    rXdist = lambda rng: rng.gamma(shape = shape1, scale =  1/rate1)

    gamma2 = scipy.stats.gamma(a = shape2, scale =  1/rate2)
    lpdfYdist = lambda x: gamma2.logpdf(x)
    rYdist = lambda rng: rng.gamma(shape = shape2, scale =  1/rate2)

    if np.allclose([shape1, rate1], [shape2, rate2]):
        X = rXdist(rng)
        Y = copy.deepcopy(X)
        isEqual = True
    else:
        X, Y, isEqual = c_maximal(DEFAULT_C, rng, rXdist, lpdfXdist, rYdist, lpdfYdist)
    return X, Y, isEqual

def c_maximal_diag_normal(rng, mu1, S1, mu2, S2):
    """
    Specialized version of c_maximal for multivariate normals 
    with diagonal covariances

    Inputs:
        rng: BitGenerator
        mu1: (D,) array, mean of first normal
        S1: (D,) array, variance of first normal

    """

    lpdfXdist = lambda x: weights.diagNormal(x, mu1, S1)
    rXdist = lambda rng: rng.normal(loc=mu1, scale=np.sqrt(S1))

    lpdfYdist = lambda x: weights.diagNormal(x, mu2, S2)
    rYdist = lambda rng: rng.normal(loc=mu2, scale=np.sqrt(S2))

    if np.allclose([mu1, S1], [mu2, S2]):
        X = rXdist(rng)
        Y = copy.deepcopy(X)
        isEqual = True
    else:
        X, Y, isEqual = c_maximal(DEFAULT_C, rng, rXdist, lpdfXdist, rYdist, lpdfYdist)
    return X, Y, isEqual