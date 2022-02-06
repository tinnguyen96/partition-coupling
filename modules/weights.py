"""
Compute probabilities of label-reassignment Gibbs moves 
"""

from scipy.special import gammaln, logsumexp

import partitions
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import ot
import igraph

## graph coloring weights ---------------------------------------------------------
def colorProbs(g, nColors, n, 
                labelList, clusts):
    """color_probs returns  probability of new color assigments that reflects the 
    distribution over partitions implied by uniform coloring
    This is NOT the uniform probability over valid colors.

    Args:
        g: igraph Graph object with N vertices
        ncolors: scalar, number of different colors in total
        n: scalar, index of node to re-color
        labelList: (N,) list of LabelHelpers
        clusts: (K,) list of sets

    Output:
        probs: (Ktilde,) vector of label assignment probabilities

    Runtime:

    """
    neighbors = igraph.Graph.neighbors(g, n)
    neighborColors = set([labelList[v].getValue() for v in neighbors])

    currPSize = len(clusts)
    nClust = min(currPSize+1, nColors) # number of clusters is capped by nColors
    legalColors = set(range(nClust)).difference(neighborColors)
    probs = np.zeros(nClust)

    logMasses = []
    for color in legalColors:
        # if n is assigned to color, how many elements are there in the partition?
        # if there are no existing vertices with color, partition size increases by 1.
        # else it stays the same.
        if (color == currPSize):
            pSize = currPSize + 1
        else:
            pSize = currPSize
        logMasses.append(-gammaln(nColors + 1 - pSize))

    # normalize probs
    normalizer = logsumexp(logMasses)
    for (idx, v) in enumerate(legalColors):
        probs[v] = np.exp(logMasses[idx]-normalizer) 
    assert np.isclose(np.sum(probs),1)
    return probs

def diagNormal(x, mu1, S1):
    """
    Inputs:
        x: (D,) array, location to evaluate normal density
        mu1: (D,) array, mean of first normal
        S1: (D,) array, variance of first normal

    Output:
        lpdf: scalar, log pdf of multivariate normal density at 
        x with mean mu1 and diagonal covariance S1
    """
    logNormConsts = -0.5*np.log(2*np.pi*S1)
    LL = -0.5*np.square(x - mu1)/S1
    lpdf = np.sum(logNormConsts) + np.sum(LL) 
    return lpdf 

## DPMM weights for Gibbs sampler ------------------------------------------------------------------
def clusterProbs(data, 
                clusts, means, clustSums,
                sd1, sd0, alpha, 
                n, rng):
    """clusterProbs computes probabilities of assignments for the Gibbs
    sampler, either using Neal 2000 Equation 3.6 (instantiating cluster means)
    or Equation 3.7 (integrating out cluster means).

    Args:
        data: numpy array of data of shape [N, D]
        clusts: list of the memberships of each cluster (list of set of
            int)
        means: (K,) list of (D,) array, the cluster mean of current partition
        clustSums: (K, D), sums of all data (coordinate-wise) in a cluster
        sd1: (D,) observation sd
        sd0: (D,) component mean sd 
        alpha: scalar, DP concentration parameter
        n: index of data-point for which to compute Gibbs conditional

    Output:
        loc_probs:

    Runtime:
        O(ND)
    """

    # dimension of the data points
    data_dim = data.shape[1]

    # cluster-specific variance and precision
    Sig1, Prec1 = sd1**2, sd1**(-2)

    # prior variance and precision
    Sig0, Prec0 = sd0**2, sd0**(-2)

    # prior mean on cluster parameters
    mu0 = np.zeros(data_dim)
    Nclust = len(clusts)  # initial number of clusters

    # unnormalized log probabilities for the clusters
    logWeights = np.zeros(Nclust + 1)
    # find the unnormalized log probabilities
    # for each existing cluster
    for c in range(Nclust):
        clust_c = list(clusts[c])
        c_size = len(clusts[c])
        
        if (means is None):
            if (clustSums is None):
                # sum all the points in this cluster
                sum_data = np.sum(data[clust_c], axis=0)
            else:
                # use pre-computed sums
                sum_data = clustSums[c]
            c_Precision = Prec0 + c_size * Prec1
            c_Sig = 1./c_Precision
            c_mean = c_Sig*(Prec1 *sum_data + Prec0*mu0)
            logWeights[c] = np.log(c_size) + diagNormal(data[n], c_mean, c_Sig + Sig1)
        else:
            # Equation 3.6
            logWeights[c] = np.log(c_size) + diagNormal(data[n], means[c], Sig1)

    # find the unnormalized log probability
    # for the "new" cluster
    logWeights[Nclust] = np.log(alpha) + diagNormal(data[n], mu0, Sig0 + Sig1)

    # transform unnormalized log probabilities
    # into probabilities
    max_weight = max(logWeights)
    logWeights = logWeights - max_weight
    loc_probs = np.exp(logWeights)
    loc_probs = loc_probs / sum(loc_probs)

    if (means is None):
        newMean = None
    else:
        newMean = rng.normal(loc=mu0, scale=sd0)
        assert len(newMean) == data_dim

    return newMean, loc_probs

def gammaPosteriorPrec(data, 
                    means, labelList, 
                    gammaShape, gammaRate, scale):
    """
    Inputs:
        data: (N,D) array
        means: (K,) list of (D,) arrays
        labelList: (N,) list
        gammaShape, gammaRate:
        scale: scalar

    Outputs:
        precShape: scalar
        precRate: (D,) array
    """
    
    z, _ = partitions.labelHelpersToLabelsAndClusts(labelList)
    N = data.shape[0]

    nClust = len(means)
    squaredNorm = np.square(means)
    squaredNormPerDim = np.sum(squaredNorm, axis=0) # (D,)
    precShape = gammaShape + nClust/2
    precRate = gammaRate + squaredNormPerDim/2

    squaredErr = np.square(data - np.array(means)[z])
    squaredErrPerDim = np.sum(squaredErr, axis=0) # (D,)
    precShape += N/2
    precRate += scale*squaredErrPerDim/2

    return precShape, precRate

def gammaPosteriorPrecIsoSd(data, 
                    means, labelList, 
                    gammaShape, gammaRate, scale):
    """
    Inputs:
        data: (N,D) array
        means: (K,) list of (D,) arrays
        labelList: (N,) list
        gammaShape, gammaRate:
        scale: scalar

    Outputs:
        precShape: scalar
        precRate: scalar
    """
    
    z, _ = partitions.labelHelpersToLabelsAndClusts(labelList)
    N, D = data.shape

    nClust = len(means)
    squaredNorm = np.square(means) # (K,D)
    squaredNormTotal = np.sum(squaredNorm)

    precShape = gammaShape + (nClust*D)/2
    precRate = gammaRate + squaredNormTotal/2

    squaredErr = np.square(data - np.array(means)[z]) # (N,D)
    squaredErrTotal = np.sum(squaredErr)

    precShape += (N*D)/2
    precRate += scale*squaredErrTotal/2

    return precShape, precRate

def GaussianPosterior(data, 
                    clusts, clustSums, 
                    sd1, sd0, c):
    """
    Compute posterior mean and covariance for the cluster mean 
    of the c cluster.
    
    Inputs:

    Outputs:
        posMean: (D,) array
        posSig: (D,) array

    Runtime:
   
    """
    
    D = data.shape[1]

    # cluster-specific variance and precision
    Sig1, Prec1 = np.square(sd1), np.square(1/sd1)
    # prior variance and precision
    Sig0, Prec0 = np.square(sd0), np.square(1/sd0)
    
    clustSize = len(clusts[c])
    posPrec = Prec0 + clustSize*Prec1
    posSig = 1./posPrec
    
    mu0 = np.zeros(D)
    
    bias = Prec0*mu0 + Prec1*clustSums[c]
    posMean = posSig*bias
    
    return posMean, posSig

## ------------------------------------------------------------------
## split-merge code
class SplitMergeMove():

    def __init__(self, 
                data, sd1, sd0, alpha,
                splitMergeDict, 
                labelList, clusts, rng):
        self.data = data
        self.sd1 = sd1 
        self.sd0 = sd0
        self.alpha = alpha 

        self.splitMergeDict = splitMergeDict
        self.rng = rng

        self.labelList = labelList
        self.clusts = clusts
        self.rng = rng
        return 

    def makeLaunchState(self, z, i, j, S, numRes):
        """
        Create launch state.

        Args:
            z: (N,) labeling of data-points, np.array of natural numbers
            i, j: scalars, indices selected for split-merge
            S: list, list, the indices of data in the same component as either i or j (excluding i, j themselves)
            numRes: scalar, number of restricted Gibbs scan

        Outputs:
            labelList: (N,) list, each is a LabelHelper object
        """
        # make initial state
        z_launch = z.copy()

        if z_launch[i] == z_launch[j]:
            # don't need to worry about consistency of z_launch since
            # the cluster that originally had i and j still has j
            z_launch[i] = max(z)+1 

        new_clusts = self.rng.binomial(1,0.5,size=len(S))
        for it in range(len(S)):
            idx = S[it]
            if (new_clusts[it] == 0):
                z_launch[idx] = z_launch[i]
            else:
                z_launch[idx] = z_launch[j]

        # restricted scan
        launchLabelList = partitions.zToLabelHelpers(z_launch)
        labelList, _ = self.proposeRestrictedGibbs(launchLabelList, 
                                                i, j, S, 
                                                numRes)
        return labelList

    def makeIJS(self, z, clusts, i=None, j=None):
        """
        Inputs:
            z: (N,) array, label-assignment
            clusts: list of sets, each set is the indices of data in one
                cluster
            i, j: (optional) scalars, if passed, will only compute S
        Outputs:
            i,j: scalars, data indices
            S: list, the indices of data in the same component as either i or j (excluding i, j themselves)
        """
        N = z.shape[0]
        if (i is None) or (j is None):
            i,j = self.rng.choice(N, 2, replace=False)
        S_as_set = (clusts[z[i]] | clusts[z[j]]) - set([i,j])
        # convert S to list to ensure iteration order is consistent
        S = list(S_as_set)
        return i, j, S

    def proposeRestrictedGibbs(self, labelList, i, j, S, numRes):
        """
        Args:
            labelList: (N,) list, each is a LabelHelper object
            i, j: scalars, indices selected for split-merge
            S: list,
            numRes: scalar, how many restricted sweeps to do

        Output: 
            nextLabelList: (N,) list
            nextClusts: (K,) list of sets, each set is the indices of data in one
                cluster

        Remark:
            this function needs to coordinate with proposal_prob in terms of the 
            ordering of the indices S, since the random MH proposal is determined
            by both the launch state but also the ordering of S which governs the 
            restricted Gibbs.
        """

        z, _ = partitions.labelHelpersToLabelsAndClusts(labelList)
        assert z[i] != z[j]

        # make copy of start state 
        nextLabelList = partitions.zToLabelHelpers(z)
        nextClusts = partitions.zToClusts(z)

        for it in range(numRes):
            for idx in S:
                # remove idx from clusts. Will not change the labels of any clusters
                # (which only happens if the popped idx were the only member of its cluster)
                # since we know that the cluster of i and j will always have at least
                # i or j
                removedClust = partitions.pop_idx(nextLabelList, nextClusts, 
                                                    None, None, 
                                                    idx, None)
                assert removedClust is None

                # z, clusts, _ = partitions.pop_idx(z, clusts, idx)
                probs, _ = self.restrictedClusterWeights(nextLabelList, nextClusts, 
                                                            idx, i, j)
                # print("it = %d" %it, "probs: ", probs)
                clust_idx = self.rng.choice(len(probs), p=probs)
                # update clusts and assignments
                if (clust_idx == 0):
                    newC = nextLabelList[i].getValue()
                else:
                    assert (clust_idx == 1)
                    newC = nextLabelList[j].getValue()
                
                labelObj = partitions.labelOfCluster(nextLabelList, nextClusts, newC)
                nextLabelList[idx] = labelObj
                nextClusts[newC].add(idx)
        return nextLabelList, nextClusts

    def logLikelihood(self, clusts, c):
        """
        Compute prod_{k} {c_k=c} int F(y_k, phi) d H_{k,c}(phi) in Jain et al Equation 3.7
        for Gaussian likelihoods and priors parametrized by self.sd1 and self.sd0. 

        Remark:
            Should change clusts to just have indices of data that are in the same 
            cluster as c. Don't need the whole partition. 

        Input:
            clusts: (K,) list of sets, each set gives the data indices in one cluster
            c: scalar, cluster index
        
        Output:
            log_L: scalar
        """

        clust = sorted(clusts[c])
        clust_size = len(clust)
        log_L = 0.0

        data_dim = self.data.shape[1]
        # cluster-specific variance and precision
        Sig1, Prec1 = self.sd1**2, self.sd1**(-2)
        # prior variance and precision
        Sig0, Prec0 = self.sd0**2, self.sd0**(-2)
        mu0 = np.zeros(data_dim)

        # print("clust ", clust)
        obs_range = []
        for it in range(clust_size):
            idx = clust[it]
            # evaluate prior
            if (it == 0): 
                term = np.sum(stats.norm.logpdf(x=self.data[idx], loc=mu0, scale=np.sqrt(Sig0 + Sig1)))
            # evaluate rolling posterior
            else:
                num_obs = len(obs_range)
                c_Precision = Prec0 + num_obs * Prec1
                c_Sig = 1./c_Precision
                sum_data = np.sum(self.data[obs_range], axis=0)
                c_mean = c_Sig*(Prec1 *sum_data + Prec0*mu0)
                term = np.sum(stats.norm.logpdf(x=self.data[idx], loc=c_mean, scale=np.sqrt(c_Sig + Sig1)))
            log_L += term
            obs_range.append(idx)

        return log_L

    def restrictedClusterWeights(self, 
                                labelList, clusts, 
                                idx, i, j):
            """
            computes restrictedClusterWeights for restricted_Gibbs

            Args:
                data: numpy array of data of shape [N, D]
                labelList: (N,) list, each is a LabelHelper object
                clusts: list of the memberships of each cluster (list of list of
                    int)
                    sampling.restricted_Gibbs should have popped idx from clusts
                sd1, sd0: observation variance and component mean variance (scalars)
                alpha: DP parameter controlling dispersion of mixing proportions
                i, j: scalars
                idx: scalar, index of data to change label

            Outputs:
                loc_probs: (2,) array, 0th entry is probability of assigning to i's
                    cluster, 1st entry assigning to j's cluster
            """
            # dimension of the data points
            data_dim = self.data.shape[1]

            # cluster-specific variance and precision
            Sig1, Prec1 = self.sd1**2, self.sd1**(-2)

            # prior variance and precision
            Sig0, Prec0 = self.sd0**2, self.sd0**(-2)

            # prior mean on cluster parameters
            mu0 = np.zeros(data_dim)

            # unnormalized log probabilities for the clusters
            log_weights = np.zeros(2)

            # cluster containing i
            iclust = list(clusts[labelList[i].getValue()])
            c_Precision = Prec0 + len(iclust) * Prec1
            c_Sig = 1./c_Precision
            # sum all the points in this cluster
            sum_data = np.sum(self.data[iclust], axis=0)
            i_mean = c_Sig*(Prec1 *sum_data + Prec0*mu0)
            log_weights[0] = np.log(len(iclust)) + diagNormal(self.data[idx], i_mean, c_Sig + Sig1)
            # log_weights[0] = np.log(len(iclust)) + np.sum(stats.norm.logpdf(
            #     x=data[idx], loc=i_mean, scale=np.sqrt(c_Sig + Sig)))

            # cluster containing j
            jclust = list(clusts[labelList[j].getValue()])
            c_Precision = Prec0 + len(jclust) * Prec1
            c_Sig = 1./c_Precision
            # sum all the points in this cluster
            sum_data = np.sum(self.data[jclust], axis=0)
            j_mean = c_Sig*(Prec1 *sum_data + Prec0*mu0)
            log_weights[1] = np.log(len(jclust)) + diagNormal(self.data[idx], j_mean, c_Sig + Sig1)
            # log_weights[1] = np.log(len(jclust)) + np.sum(stats.norm.logpdf(
                # x=data[idx], loc=j_mean, scale=np.sqrt(c_Sig + Sig)))

            # transform unnormalized log probabilities
            # into probabilities
            max_weight = max(log_weights)
            log_weights = log_weights - max_weight
            probs = np.exp(log_weights)
            norm_const = sum(probs)
            log_probs = np.log(probs) - np.log(norm_const)
            final_probs = probs / norm_const
            # print("probs is ", probs, "log_probs is ", log_probs)
            return final_probs, log_probs

    def proposalProbability(self,  
                            startLabelList, 
                            endLabelList,
                            i, j, S):
        """
        Compute the probability of transitioning from z_start to z_end using 
        restricted_Gibbs. There is only one way of taking restricted_Gibbs move
        to end at z_split when beginning from z_launch, if the list S has been fixed.

        Inputs:
            startLabelList:
            endLabelList:
            i, j, S:

        Outputs:
            cumu_log_prob: scalar

        Remark:
            Jain and Neal 2004 Eq 3.14 doesn't explicitly say what 
            the ordering of the indices in S is. Wallach's code seemingly uses a for 
            comprehension for S (line 59 in conjugate_split_merge.py), but it looks like 
            they compute the z_split and the proposal in the same loop, so that ensures 
            the same ordering of the indices in S.
        """

        startZ, _ = partitions.labelHelpersToLabelsAndClusts(startLabelList)

        assert startZ[i] != startZ[j]

        # print("startZ", startZ, "z_end", z_end)

        # make copy of start state 
        startLabelListCopy = partitions.zToLabelHelpers(startZ)
        startClustsCopy = partitions.zToClusts(startZ)

        cumu_log_prob = 0.0

        for idx in S:
            # print("z, clusts before pop", z, clusts)
            # z, clusts, _ = partitions.pop_idx(z, clusts, idx)
            removedClust =  partitions.pop_idx(startLabelListCopy, startClustsCopy, None, None, idx, None)
            assert removedClust is None
            # print("clusts after pop", clusts)
            _, log_probs = self.restrictedClusterWeights(startLabelListCopy, startClustsCopy, 
                                                        idx, i, j)
            final_val = endLabelList[idx].getValue()
            # print("final_val ",final_val)
            if (final_val == endLabelList[i].getValue()):
                cumu_log_prob += log_probs[0]
            else:
                assert final_val == endLabelList[j].getValue()
                cumu_log_prob += log_probs[1]

            # if (final_val == len(startClustsCopy)):
            #     startClustsCopy.append(set())
            #     labelObj = partitions.LabelHelper(final_val)
            # else:
            #     labelObj = partitions.labelOfCluster(startLabelListCopy, startClustsCopy, final_val)

            labelObj = partitions.labelOfCluster(startLabelListCopy, startClustsCopy, final_val)
            startLabelListCopy[idx] = labelObj
            startClustsCopy[final_val].add(idx)
            # print("idx ", idx, "cumu_log_prob after idx", cumu_log_prob)

        return cumu_log_prob

    def acceptanceProbability(self, 
                                labelList, clusts,
                                launchLabelList,
                                nextLabelList, nextClusts,
                                i, j, S):
        """
        Compute log acceptance probability. 

        Inputs:
            labelList: (N,) list, current label assignments
            clusts: (K,) list of sets
            z_launch: (N,) array, label assignments from make_launch
            nextLabelList: (N,) array, candiate next assignment 
            i,j: scalars,
            S: list, data indices in either z[i]'s' or z[j]'s cluster

        Output:
            log_acc: scalar

        Remark:
            For efficiency should save clusts and clusts_next from other calls.
        """

        # split 
        if (labelList[i].getValue() == labelList[j].getValue()):
            clust_i = clusts[labelList[i].getValue()]
            next_clust_i, next_clust_j = nextClusts[nextLabelList[i].getValue()], nextClusts[nextLabelList[j].getValue()]
            log_proposal_ratio = -self.proposalProbability(launchLabelList,
                                                            nextLabelList, 
                                                            i, j, S)
            log_prior_ratio = np.log(self.alpha) + gammaln(len(next_clust_i)) + \
                             gammaln(len(next_clust_j)) - gammaln(len(clust_i))
            log_likelihood_ratio = 0.0
            log_likelihood_ratio += self.logLikelihood(nextClusts, nextLabelList[i].getValue())
            log_likelihood_ratio += self.logLikelihood(nextClusts, nextLabelList[j].getValue())
            log_likelihood_ratio -= self.logLikelihood(clusts, labelList[i].getValue())
        # merge
        else:
            next_clust_i = nextClusts[nextLabelList[i].getValue()]
            clust_i = clusts[labelList[i].getValue()]
            clust_j = clusts[labelList[j].getValue()]

            log_proposal_ratio = self.proposalProbability(launchLabelList, 
                                                    labelList, 
                                                    i, j, S)
            log_prior_ratio = -np.log(self.alpha) + gammaln(len(next_clust_i)) - \
                            gammaln(len(clust_i)) - gammaln(len(clust_j))
            log_likelihood_ratio = 0.0
            log_likelihood_ratio += self.logLikelihood(nextClusts, nextLabelList[i].getValue())
            log_likelihood_ratio -= self.logLikelihood(clusts, labelList[i].getValue())
            log_likelihood_ratio -= self.logLikelihood(clusts, labelList[j].getValue())

        if (False):
            print("log_proposal_ratio ", log_proposal_ratio)
            print("log_prior_ratio ", log_prior_ratio)
            print("log_likelihood_ratio ", log_likelihood_ratio)

        logAcc = log_proposal_ratio + log_prior_ratio + log_likelihood_ratio

        logAcc = np.amin([0.0, logAcc]) # acceptance ratio might be more than 1
        return logAcc

    def sample(self, i, j):
        """
        Inputs:
            i, j: scalars (or None), indices of data to localize clusters for split/merge
                if None, decide i, j in this call
        
        Output:
            Propose either a split or a merge and decide if we switch to that state. 

        """
        z, clusts = partitions.labelHelpersToLabelsAndClusts(self.labelList)

        # make launch state
        if (i is None) or (j is None):
            i, j, S = self.makeIJS(z, clusts)
        # print("clusts ", clusts)
        # print("i ", i, "j ", j)
        # print("S ", S)
        else:
            S_as_set = (clusts[z[i]] | clusts[z[j]]) - set([i,j])
            # convert S to list to ensure iteration order is consistent
            S = list(S_as_set)

        launchLabelList = self.makeLaunchState(z, i, j, S, self.splitMergeDict["numRes"])
        # print("z_launch ", z_launch)

        # make split proposal
        if (z[i] == z[j]):
            # print("Proposing split")
            nextLabelList, nextClusts = self.proposeRestrictedGibbs(launchLabelList, i, j, S, 1)
        # make merge proposal
        else:
            # print("Proposing merge")
            # probably don't actually need to get the merged state to compute the 
            # acceptance probability
            nextLabelList, _, nextClusts = partitions.reassignClust(z, clusts, z[i], z[j])
        # print("Proposed state is z_next ", z_next)

        logAcc = self.acceptanceProbability(self.labelList, self.clusts,
                                                launchLabelList,
                                                nextLabelList, nextClusts,
                                                i, j, S)
        # print("log acc is ", logAcc)

        # accept or reject
        if (np.log(self.rng.uniform()) < logAcc):
            # print("Accept proposal")
            accept = 1
            labelList = nextLabelList
            clusts = nextClusts
        else:
            # print("Reject proposal")
            accept = 0
            labelList = self.labelList
            clusts = self.clusts

        return accept, labelList, clusts

## deprecated code ------------------------------------------------------------------

# def fastWeights(data, 
#             clusts, means, clustSums,
#             sd1, sd0, alpha, 
#             n, rng):
#     """fastWeights computes probabilities of assignments for the Gibbs
#     sampler, either using Neal 2000 Equation 3.6 (instantiating cluster means)
#     or Equation 3.7 (integrating out cluster means).

#     Args:
#         data: numpy array of data of shape [N, D]

#         clusts: list of the memberships of each cluster (list of set of
#             int)
#         means: (K,) list of (D,) array
#         clustSums: (K, D), sums of all data (coordinate-wise) in a cluster

#         sd1: (D,) observation sd
#         sd0: (D,) component mean sd 
#         alpha: scalar, DP concentration parameter

#         n: index of data-point for which to compute Gibbs conditional

#     Output:

#     Runtime:
#         O(KD)
#     """
#     # dimension of the data points
#     D = data.shape[1]

#     # cluster-specific variance and precision
#     Sig1, Prec1 = np.square(sd1), np.square(1/sd1)
#     # prior variance and precision
#     Sig0, Prec0 = np.square(sd0), np.square(1/sd0)

#     # prior mean on cluster parameters is zero
#     mu0 = np.zeros(D)
#     Nclust = len(clusts)  # initial number of clusters

#     # unnormalized log probabilities for the clusters
#     log_weights = np.zeros(Nclust + 1)

#      # find the unnormalized log probability
#     # for the "new" cluster
#     log_weights[Nclust] = np.log(alpha) + diagNormal(data[n], mu0, Sig0 + Sig1)

#     # find the unnormalized log probabilities
#     # for each existing cluster
#     for c in range(Nclust):
#         c_size = len(clusts[c])

#         if (means is None):
#             # Equation 3.7
#             c_Precision = Prec0 + c_size * Prec1
#             c_Sig = 1./c_Precision
#             c_mean = c_Sig*(Prec1*clustSums[c] + Prec0*mu0) # all vectors are (D,) so this should be entry-wise multiplication
#             log_weights[c] = np.log(c_size) + diagNormal(data[n], c_mean, c_Sig + Sig1)
#         else:
#             # Equation 3.6
#             log_weights[c] = np.log(c_size) + diagNormal(data[n], means[c], Sig1)

#     # transform unnormalized log probabilities
#     # into probabilities
#     max_weight = max(log_weights)
#     log_weights = log_weights - max_weight
#     loc_probs = np.exp(log_weights)
#     loc_probs = loc_probs / sum(loc_probs)

#     if (means is None):
#         newMean = None
#     else:
#         newMean = rng.normal(loc=mu0, scale=sd0)
#         assert len(newMean) == D

#     return newMean, loc_probs

# # track cluster means
# def fastWeights_withMeans(data, 
#                         means, clusts, 
#                         sd1, sd0, alpha, n, rng):
#     """
   
#     """

#     # dimension of the data points
#     data_dim = data.shape[1]

#     # cluster-specific variance and precision
#     Sig, Prec = sd**2, sd**(-2)

#     # prior variance and precision
#     Sig0, Prec0 = sd0**2, sd0**(-2)

#     # prior mean on cluster parameters
#     mu0 = np.zeros(data_dim)
#     Nclust = len(clusts)  # initial number of clusters
    
#     # unnormalized log probabilities for the clusters
#     log_weights = np.zeros(Nclust + 1)
    
#     # find the unnormalized log probabilities
#     # for each existing cluster
#     for c in range(Nclust):
#         c_size = len(clusts[c])
#         log_weights[c] = np.log(c_size) + np.sum(stats.norm.logpdf(
#             x=data[n], loc=means[c], scale=np.sqrt(Sig)))

#     # find the unnormalized log probability
#     # for the "new" cluster
#     log_weights[Nclust] = np.log(alpha) + np.sum(stats.norm.logpdf(
#         x=data[n], loc=mu0, scale=np.sqrt(Sig0 + Sig)))
    
#     max_weight = max(log_weights)
#     log_weights = log_weights - max_weight
#     loc_probs = np.exp(log_weights)
#     loc_probs = loc_probs / sum(loc_probs)
#     prior_mean = rng.normal(mu0, scale=sd0) # should be faster than multivariate_normal
#     return prior_mean, loc_probs

# def old_color_probs(g, ncolors, n, labelList):
#     """color_probs returns  probability of new color assigments that reflects the 
#     distribution over partitions implied by uniform coloring
#     This is NOT the uniform probability over valid colors.
    
#     Args:
#         g: igraph Graph object
#         ncolors: scalar, number of different colors
#         n: scalar, index of node to re-color
#         labelList: array of indices of current colors

#     Output:
#         probs: (ncolors,) vector of label assignment

#     """
#     legal = np.ones(ncolors)
#     neighbors = igraph.Graph.neighbors(g, n)

#     neighborColors = [labelList[v].getValue() for v in neighbors]

#     legal[list(set(neighborColors))] = 0.

#     valid_colors = np.nonzero(legal)[0]
#     probs = np.zeros(ncolors)

#     log_masses = []
#     # print("valid_colors", valid_colors)
#     p_sizes = []
#     for v in valid_colors:
#         # if color vertex n with v, what is the size of the implied partition
#         new_colors = labelList.copy()
#         new_colors[n] = partitions.LabelHelper(v)
#         p_size = len(partitions.labelHelpersToLabelsAndClusts(new_colors)[1])
#         p_sizes.append(p_size)
#         log_mass = -gammaln(ncolors + 1 - p_size)
#         # print("log_mass", gammaln(ncolors + 1 - p_size))
#         # print("ncolors - p_size", ncolors - p_size)
#         log_masses.append(log_mass)
#     # print("p_size", p_sizes)
#     # print("log_masses", log_masses)

#     # normalize probs
#     normalizer = logsumexp(log_masses)
#     for (idx, v) in enumerate(valid_colors):
#         probs[v] = np.exp(log_masses[idx]-normalizer) 
#     assert np.isclose(np.sum(probs),1)
#     return probs

# def gammaPosteriorClusterPrec(data, 
#                     means, labelList, 
#                     gammaShape, gammaRate):

#     """
#     Inputs:
#         data: (N,D) array
#         means: (K,) list of (D,) arrays
#         labelList: (N,) list

#     Outputs:
        
#         clusterPrecShape: scalar
#         clusterPrecRate: (D,) array
#     """

#     z, _ = partitions.labelHelpersToLabelsAndClusts(labelList)
#     N = data.shape[0]

#     nClust = len(means)
#     squaredNorm = np.square(means)
#     squaredNormPerDim = np.sum(squaredNorm, axis=0) # (D,)
#     clusterPrecShape = gammaShape + nClust/2
#     clusterPrecRate = gammaRate + squaredNormPerDim/2

#     return clusterPrecShape, clusterPrecRate

# def gammaPosteriorNoisePrec(data, 
#                     means, labelList, 
#                     gammaShape, gammaRate):
#     """
#     noisePrecShape: scalar,
#     noisePrecRate: (D,) array
#     """

#     z, _ = partitions.labelHelpersToLabelsAndClusts(labelList)
#     N = data.shape[0]

#     squaredErr = np.square(data - np.array(means)[z])
#     squaredErrPerDim = np.sum(squaredErr, axis=0) # (D,)
#     noisePrecShape = gammaShape  + N/2
#     noisePrecRate = gammaRate  + squaredErrPerDim/2

#     return noisePrecShape, noisePrecRate
