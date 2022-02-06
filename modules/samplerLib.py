# Todos:

# enforece that labelList has the right invariant (all i such that labelList[i] have the same value, need to check by reference)

# move the reassignClust() function into the paritions.py module

# make the split merge code compatible with the label vector being objects rather than np.array vectors

##    add function in MCMCSingle to check invariant that all label values are contiguous from 0 to K-1 for some K.
# check that notebooks like gene and 2DStandardize still work with the subclass DPMMSingle()

import math
import pandas as pd
import logging
import time
import numpy as np
from scipy.special import gammaln

import partitions, weights, priors

import couplings

# global variables 
DEFAULT_BURNIN_RATIO=0.1
TRUTH_NREP=10

# DPMM 
PRIOR_SD_GAMMA_SHAPE=0.1
PRIOR_SD_GAMMA_RATE=0.1
PRIOR_ALPHA_GAMMA_SHAPE=2
PRIOR_ALPHA_GAMMA_RATE=4

## graph coloring
SMALLEST_NSTEP=500
PRINT_INTERVAL=1000

# DEBUG
DEBUG=False

class MCMC():
    def __init__(self, 
                timeBudget,
                hFunctionDict,
                rng,
                **kwargs):
        self.rng = rng
        self.hFunctionDict = hFunctionDict
        self.remainingTime = timeBudget
        return 

    def estimate(self, burnIn, nIter):
        raise NotImplementedError

    def getTimeTaken(self):
        self.timeTaken = self.setUpTime + self.estimateTime
        return self.timeTaken

class MCMCSingle(MCMC):

    # def __init__(self, 
    #             data, 
    #             initDict, hyperPriorDict, 
    #             sampleNonPartitionList, 
    #             timeBudget, hFunctionDict, rng):

    def __init__(self, 
                timeBudget, 
                hFunctionDict, 
                rng,
                **kwargs):
        MCMC.__init__(self,
                timeBudget, 
                hFunctionDict, 
                rng,
                **kwargs)
        return 

    def setInitialState(self, initDict):
        raise NotImplementedError

    def sampleAssignment(self, n):
        raise NotImplementedError

    def samplePartition(self):
        raise NotImplementedError

    def sampleNSteps(self, nStep):
        raise NotImplementedError

    def getState(self):
        raise NotImplementedError

    def getEvaluation(self):
        return self.h.copy()

    def checkInvariants(self):
        # check that the labels in self.labelList are contiguous values from 0 to K -1 
        z, _ = partitions.labelHelpersToLabelsAndClusts(self.labelList)
        minZ, maxZ = np.amin(z), np.amax(z)
        assert np.allclose(np.unique(z), np.arange(minZ, maxZ+1))

        # check that self.clusts[i] are all the data points n where self.labelList[n].getValue() 
        # is equal to i.
        nClust = len(self.clusts)
        for i in range(nClust):
            dataIndices = self.clusts[i]
            values = [self.labelList[n].getValue() for n in dataIndices]
            assert np.allclose(values, i*np.ones(len(dataIndices)))
        return 

    def getMovingAverage(self):
        cumSum = np.cumsum(self.h, axis=0)
        norms = np.arange(1, len(self.h) + 1)
        movingAverage = cumSum/norms[:,np.newaxis]
        estNames = ["est%d" %d for d in range(self.hFunctionDict["nHColumns"])]
        df = pd.DataFrame(data=movingAverage, columns=estNames)
        return df

    def evaluateHFunction(self):
        """
        Runtime:
            O(time to evaluate function on partition of size N)
        """
        value = self.hFunctionDict["hFunction"](self.labelList, self.clusts)
        self.h.append(value)
        return

    def estimate(self, burnIn, maxIter):
        """
        Runtime:
            O(maxIter x NKD)
        """
        st = time.process_time()
        
        if (maxIter == -1):
            assert (burnIn == -1) and np.isfinite(self.remainingTime)
            while (True):
                self.sampleNSteps(1)
                timeTaken = time.process_time() - st
                if (timeTaken > self.remainingTime):
                    break
            burnIn = int(len(self.h)*DEFAULT_BURNIN_RATIO)

        else:
            self.sampleNSteps(maxIter)

        burnIns = []
        assert len(self.h) >= burnIn

        ests = self.h[burnIn:] 

        timeAverage = np.mean(ests, axis=0)
        lastIterate = np.array(self.h[-1])

        estInfo = {"timeAverage": timeAverage, "lastIterate": lastIterate}
        timeInfo = {"nStep": len(self.h), "tau": -1, "hasMet": True, "nXstep": len(self.h)}
        auxInfo = None
        
        self.estimateTime = time.process_time() - st
        return estInfo, timeInfo, auxInfo

class DPMMSingle(MCMCSingle):
    # def __init__(self, 
    #             data, 
    #             initDict, hyperPriorDict, 
    #             fastAssignment,
    #             sampleNonPartitionList, 
    #             timeBudget, hFunctionDict, rng):

    def __init__(self, 
                timeBudget, 
                hFunctionDict, 
                rng,
                **kwargs):
        """
        Inputs:
            data: (N,D) array
            initDict: dict
            hyperPriorDict:
            fastAssignment: boolean
            sampleNonPartitionList: list:
            timeBudget: scalar, time budget in seconds
            hFunctionDict: 
            rng: BitGenerator
        """

        MCMCSingle.__init__(self, timeBudget, hFunctionDict, rng)

        self.data = kwargs["data"]
        self.N, self.D = self.data.shape
        
        self.hyperParamsTrace = []
        self.setUpTime = self.setInitialState(kwargs["initDict"])

        self.hyperPriorDict = kwargs["hyperPriorDict"]
        self.fastAssignment = kwargs["fastAssignment"]
        self.sampleNonPartitionList = kwargs["sampleNonPartitionList"]

        self.estimateTime = 0

        return 

    def setInitialState(self, initDict):

        st = time.process_time()

        # DP hyper-params
        self.sd1 = initDict["sd1"]
        self.sd0 = initDict["sd0"]
        self.alpha = initDict["alpha"]

        # partition and cluster means
        res = priors.initialPartitionAndMeans(self.data, 
                                    initDict["initZ"], 
                                    self.alpha, 
                                    initDict["initType"], 
                                    self.rng)
        self.labelList, self.clusts, self.clustSums, self.means = res # clustSums is auxiliary data structures (saves compute time)

        self.h = []
        self.evaluateHFunction()
        self.accumulateHyperParam()

        timeTaken = time.process_time() - st
        return timeTaken

    def computeAuxiliary(self):
        """
        Using self.labelList and self.clusts, set clustSums (and a dummy means)
        """
        nLabels = len(self.clusts)
        clustSums, means = [0 for _ in range(nLabels)], [0 for _ in range(nLabels)]
        for label in range(nLabels):
            indices = list(self.clusts[label])
            clustSums[label] = np.sum(self.data[indices,:], axis=0)
            means[label] = np.mean(self.data[indices,:], axis=0)
        self.clustSums = clustSums
        self.means = means
        return 

    def sampleAssignment(self, n):
        """
        Runtime:
            O(KD)
        """
        _ = partitions.pop_idx(self.labelList, 
                              self.clusts, self.means, self.clustSums, n, self.data[n,:])

        if ("means" in self.sampleNonPartitionList):
            thisMeans = self.means
        else:
            thisMeans = None

        if (self.fastAssignment):
            thisClustSums = self.clustSums
        else:
            thisClustSums = None

        newMean, loc_probs = weights.clusterProbs(self.data, 
                                    self.clusts, thisMeans, thisClustSums,
                                    self.sd1, self.sd0, self.alpha, 
                                        n, self.rng)

        # the data point will be assigned to the newz cluster
        newz = self.rng.choice(len(loc_probs), p=loc_probs) 

        # if necessary, instantiate a new cluster
        if newz == len(self.clusts): 
            labelObj = partitions.LabelHelper(newz)
            self.clusts.append(set())
            self.means.append(newMean)
            self.clustSums.append(np.zeros(self.D))
        else:
            labelObj = partitions.labelOfCluster(self.labelList, self.clusts, newz)

        # update cluster assigments
        self.clusts[newz].add(n)
        self.labelList[n] = labelObj
        self.clustSums[newz] = self.clustSums[newz] + self.data[n,:]
        return 

    def samplePartition(self):
        raise NotImplementedError

    def sampleConcentration(self):
        """
        Use gamma prior and augmentation scheme as in Escobar and West 1995
        Section 6 to sample DP concentration parameter. 
        """
        eta = self.rng.beta(a=self.alpha+1, b=self.N)
        nClust = len(self.clusts)

        gammaShape = PRIOR_ALPHA_GAMMA_SHAPE + nClust - 1
        gammaRate = PRIOR_ALPHA_GAMMA_RATE - np.log(eta)

        ratio = (gammaShape)/(self.N*gammaRate)
        piEta = ratio/(ratio + 1)

        mixt = self.rng.binomial(1,piEta)
        gammaShape += mixt
        self.alpha = self.rng.gamma(shape=gammaShape, scale=1/gammaRate)

        return 

    def sampleStandardDeviation(self):

        if (self.hyperPriorDict["separateSd"]):

            precShape, precRate  = weights.gammaPosteriorPrec(self.data, 
                                        self.means, self.labelList, 
                                        PRIOR_SD_GAMMA_SHAPE, PRIOR_SD_GAMMA_RATE,
                                        (1/self.hyperPriorDict["sd1OverSd0"]**2))

            prec = self.rng.gamma(shape=precShape, scale=1/precRate)

            self.sd0 = np.sqrt(1/prec)
            self.sd1 = self.hyperPriorDict["sd1OverSd0"]*self.sd0

        else:
            precShape, precRate  = weights.gammaPosteriorPrecIsoSd(self.data, 
                                        self.means, self.labelList, 
                                        PRIOR_SD_GAMMA_SHAPE, PRIOR_SD_GAMMA_RATE,
                                        (1/self.hyperPriorDict["sd1OverSd0"]**2))
            prec = self.rng.gamma(shape=precShape, scale=1/precRate)

            self.sd0 = np.sqrt(1/prec)*np.ones(self.D)
            self.sd1 = self.hyperPriorDict["sd1OverSd0"]*self.sd0

        return 

    def sampleMeans(self):
        nClust = len(self.clusts)
        for c in range(nClust):
            posMean, posSig = weights.GaussianPosterior(self.data, 
                                                                self.clusts, self.clustSums, 
                                                                self.sd1, self.sd0, c)
            self.means[c] = self.rng.normal(loc=posMean, scale=np.sqrt(posSig)) # avoid using pos_cov since multivariate_normal is slow
            assert len(self.means[c]) == self.D
        return 

    def sampleNSteps(self, nStep):
        for it in range(nStep):
            self.samplePartition()
            if ("means" in self.sampleNonPartitionList):
                self.sampleMeans()

                if ("sd" in self.sampleNonPartitionList):
                    self.sampleStandardDeviation()

                if ("concentration" in self.sampleNonPartitionList):
                    self.sampleConcentration()

            # save trace/check invariantsif run experiment without time budget
            if (not np.isfinite(self.remainingTime)): 
                self.accumulateHyperParam()
                self.checkInvariants()

            self.evaluateHFunction()

            if (nStep > SMALLEST_NSTEP):
                if (it + 1) % PRINT_INTERVAL == 0:
                    print("Finished %d iterations of single chains" %it)

        state = self.getState()
        return state

    def getState(self):
        """
        Return copy of states.
        """
        hyperParams = [self.alpha, self.sd0.copy(), self.sd1.copy()]
        partition = [self.labelList.copy(), self.clusts.copy(), self.clustSums.copy(), self.means.copy()]
        state = [hyperParams, partition]
        return state

    def getTrace(self):
        """
        Return trace of hyperparameters and of function evaluation as dataframe.
        Only meant to use this if there is no limit on runtime.
        """

        assert not np.isfinite(self.remainingTime)

        sd0names = ["sd0_%d" %d for d in range(self.D)]
        sd1names = ["sd1_%d" %d for d in range(self.D)]
        names = ["nClusts", "alpha"] + sd0names + sd1names
        df = pd.DataFrame(data=self.hyperParamsTrace, columns=names)

        estNames = ["est%d" %d for d in range(self.hFunctionDict["nHColumns"])]
        hDF = pd.DataFrame(data=self.h, columns=estNames)

        assert df.shape[0] == hDF.shape[0]
        finalDF = pd.concat([df, hDF], axis=1)
        return finalDF

    def accumulateHyperParam(self):
        """
        Save trace of number of clusters and hyperparameters.
        """
        states = np.concatenate([[len(self.clusts), self.alpha], self.sd0, self.sd1])
        self.hyperParamsTrace.append(states)
        return 

class DPMMSingleGibbs(DPMMSingle):

    def __init__(self, 
                timeBudget,
                hFunctionDict, 
                rng,
                **kwargs):
        DPMMSingle.__init__(self, 
                            timeBudget,
                            hFunctionDict, 
                            rng, 
                            **kwargs)

        return 

    def samplePartition(self):
        """
        Runtime:
            O(NKD)
        """
        for n in range(self.N):
            self.sampleAssignment(n)
        return 

class DPMMSingleSplitMerge(DPMMSingle):

    def __init__(self, 
                timeBudget,
                hFunctionDict, 
                rng,
                **kwargs):

        self.splitMergeDict = kwargs["splitMergeDict"]
        DPMMSingle.__init__(self, 
                            timeBudget,
                            hFunctionDict, 
                            rng, 
                            **kwargs)

    def samplePartition(self):
        accept_list = []
        for _ in range(self.splitMergeDict["numProp"]):
            smObj = weights.SplitMergeMove(self.data, self.sd1, self.sd0, self.alpha,
                                         self.splitMergeDict, 
                                         self.labelList, self.clusts,
                                         self.rng)
            accept, self.labelList, self.clusts = smObj.sample(None, None)
            self.computeAuxiliary()
            accept_list.append(accept)

        for _ in range(self.splitMergeDict["numScan"]):
            for n in range(self.N):
                self.sampleAssignment(n)

        return accept_list

class GraphSingle(MCMCSingle):

    def __init__(self, 
                timeBudget,
                hFunctionDict, 
                rng,
                **kwargs):

        MCMCSingle.__init__(self, timeBudget, hFunctionDict, rng)

        self.graph = kwargs["graph"]
        self.nColors = kwargs["initDict"]["nColors"]
        self.N = self.graph.vcount()

        self.setUpTime = self.setInitialState(kwargs["initDict"])
        self.estimateTime = 0

        return 

    def setInitialState(self, initDict):
        st = time.process_time()

        if initDict["initType"] == "greedy":
            possible, labelList, clusts = priors.greedyColoring(self.graph, self.nColors)
            if (not possible):
                raise RuntimeError("Greedy coloring using %d colors doesn't work" %self.nColors)
            self.labelList, self.clusts = labelList, clusts
        else:
            assert initDict["initType"] is None
            self.labelList = partitions.zToLabelHelpers(initDict["initZ"])
            _, self.clusts = partitions.labelHelpersToLabelsAndClusts(self.labelList)

        self.h = []
        self.evaluateHFunction()
        timeTaken = time.process_time() - st
        return timeTaken

    def sampleAssignment(self, n):
        _ = partitions.pop_idx(self.labelList, 
                        self.clusts, None, None, 
                        n, None)

        probs = weights.colorProbs(self.graph, self.nColors, n, 
                                   self.labelList, self.clusts)
        newz = self.rng.choice(self.nColors, p=probs)

        # if necessary, instantiate a new cluster
        if (newz == len(self.clusts)):
            labelObj = partitions.LabelHelper(newz)
            self.clusts.append(set())
        else:
            labelObj = partitions.labelOfCluster(self.labelList, self.clusts, newz)

        self.labelList[n] = labelObj
        self.clusts[newz].add(n)
        return 

    def getState(self):
        """
        Return copy of states.
        """
        state = [self.labelList.copy(), self.clusts.copy()]
        return state

    def samplePartition(self):
        for n in range(self.N): 
            self.sampleAssignment(n)
        return 

    def sampleNSteps(self, nStep):
        for it in range(nStep):
            self.samplePartition()
            self.evaluateHFunction()

            if (nStep > SMALLEST_NSTEP):
                if (it + 1) % PRINT_INTERVAL == 0:
                    print("Finished %d iterations of single chains" %it)

        state = self.getState()
        return state

class MCMCCoupled(MCMC):

    def __init__(self, 
                timeBudget,
                hFunctionDict,
                rng,
                **kwargs):

        self.couplingDict = kwargs["couplingDict"]
        MCMC.__init__(self,
                timeBudget,
                hFunctionDict,
                rng)
        return 

    def areChainsEqual(self):
        raise NotImplementedError

    def setInitialState(self, initDict):
        raise NotImplementedError

    def advanceX(self):
        raise NotImplementedError

    def sampleNSteps(self, nStep):
        raise NotImplementedError

    def setUp(self, initDict):
        st = time.process_time()
        self.setInitialState(initDict)
        self.advanceX()
        self.setIntersectionSizes()
        self.distanceBetweenChains()
        self.checkInvariants()
        # print("self.XClusts after setUp", self.XClusts)
        # print("self.YClusts after setUp", self.YClusts)
        timeTaken = time.process_time() - st
        return timeTaken

    def setIntersectionSizes(self):
        """
        Runtime:
            O(KM), assuming constant time to compute set intersections
        """
        sizes = [[len(c1.intersection(c2)) for c2 in self.YClusts] for c1 in self.XClusts]
        self.intersectionSizes = np.array(sizes)
        return 

    def evaluateHFunction(self, addX, addY):
        """
        Runtime:
            O(time to evaluate function on partition of size N)
        """
        if (addX):
            XValue = self.hFunctionDict["hFunction"](self.XLabelList, self.XClusts)
            self.hX.append(XValue)
        if (addY):
            YValue = self.hFunctionDict["hFunction"](self.YLabelList, self.YClusts)
            self.hY.append(YValue)
        return

    def estimate(self, burnIn, minIter):
        """
        Inputs:
            burnIn: scalar, number of burnin iterations
            minIter: scalar, minimum number of steps

        Outputs:
            estInfo: dict
            timeInfo: dict
            auxInfo: dict
        """
        st = time.process_time()
        Xstep = 1; Ystep = 0; hasMet = False; tau = -1
        # print("\n self.XClusts in estimate", self.XClusts)
        # print("\n self.YClusts in estimate", self.YClusts)
        while True:
            if self.areChainsEqual():
                if (not hasMet):
                    hasMet = True
                    tau = Xstep
                if Xstep >= minIter: break
                else:
                    self.advanceX()
                    Xstep += 1
            else:
                # until chains have met, run double transition
                self.sampleNSteps(1)
                Ystep +=1
                Xstep += 1

            # if time has run out, we can't update the states,
            # although this is unlikely for the first meeting experiment
            # in a replicate
            time_elapsed = time.process_time()-st
            # print("time_elapsed",time_elapsed)
            if (time_elapsed >= self.remainingTime): break

        # used up compute time without meeting or if we haven't evolved enough
        # sweeps

        timeInfo = {"tau": tau, "nStep": Xstep + Ystep, "hasMet": hasMet, "nXstep": Xstep}

        auxInfo = {"dists": self.dists}

        timeAverage = extractUnbiasedEstimate(self.hFunctionDict["nHColumns"], burnIn, minIter, self.hX, self.hY, tau)

        estInfo = {"timeAverage": timeAverage}

        self.estimateTime = time.process_time()-st

        return estInfo, timeInfo, auxInfo

class DPMMCoupled(MCMCCoupled):

    def __init__(self, 
                timeBudget, 
                hFunctionDict, 
                rng,
                **kwargs):
        """ 
        
        """

        MCMCCoupled.__init__(self,
                timeBudget, 
                hFunctionDict, 
                rng,
                **kwargs)

        self.data = kwargs["data"]
        self.N, self.D = self.data.shape

        self.fastAssignment = kwargs["fastAssignment"]

        self.sampleNonPartitionList = kwargs["sampleNonPartitionList"]

        self.hyperPriorDict = kwargs["hyperPriorDict"]

        self.equal = {"partition": False, "concentration": True, "sd": True, "means": True}
        if ("sd" in self.sampleNonPartitionList):
            self.equal["sd"] = False
        if ("means" in self.sampleNonPartitionList):
            self.equal["means"] = False
        if ("concentration" in self.sampleNonPartitionList):
            self.equal["concentration"] = False

        self.dists = {"partition": [], "concentration": [], "sd": [], "means": []}

        self.setUpTime = self.setUp(kwargs["initDict"])
        self.remainingTime = self.remainingTime - self.setUpTime
        self.estimateTime = 0

        return

    def computeAuxiliary(self):
        """
        Using self.labelList and self.clusts, set clustSums (and a dummy means)
        """

        # update X 
        nLabels = len(self.XClusts)
        clustSums, means = [0 for _ in range(nLabels)], [0 for _ in range(nLabels)]
        for label in range(nLabels):
            indices = list(self.XClusts[label])
            clustSums[label] = np.sum(self.data[indices,:], axis=0)
            means[label] = np.mean(self.data[indices,:], axis=0)
        self.XClustSums = clustSums
        self.XMeans = means

        # update Y
        nLabels = len(self.YClusts)
        clustSums, means = [0 for _ in range(nLabels)], [0 for _ in range(nLabels)]
        for label in range(nLabels):
            indices = list(self.YClusts[label])
            clustSums[label] = np.sum(self.data[indices,:], axis=0)
            means[label] = np.mean(self.data[indices,:], axis=0)
        self.YClustSums = clustSums
        self.YMeans = means

        return 

    def setInitialState(self, initDict):
        """
        Runtime:
            O(ND)
        """

        # DP hyper-params
        self.Xsd1 = initDict["sd1"]
        self.Xsd0 = initDict["sd0"]
        self.Xalpha = initDict["alpha"]

        self.Ysd1 = initDict["sd1"]
        self.Ysd0 = initDict["sd0"]
        self.Yalpha = initDict["alpha"]

        # partitions
        Xres = priors.initialPartitionAndMeans(self.data, 
                                    initDict["initZ"], 
                                    self.Xalpha, 
                                    initDict["initType"], 
                                    self.rng)
        self.XLabelList, self.XClusts, self.XClustSums, self.XMeans = Xres

        Yres = priors.initialPartitionAndMeans(self.data, 
                                    initDict["initZ"], 
                                    self.Yalpha, 
                                    initDict["initType"], 
                                    self.rng)
        self.YLabelList, self.YClusts, self.YClustSums, self.YMeans = Yres

        # function values along Markov chain
        self.hX, self.hY = [], []
        self.evaluateHFunction(True, True)

        return 

    def distanceBetweenChains(self):
        self.distanceBetweenPartitions()

        if ("means" in self.sampleNonPartitionList):
            pairings, _, _ = partitions.find_pairings(self.XClusts, self.YClusts, self.intersectionSizes)
            self.distanceBetweenMeans(pairings)
            
            if ("sd" in self.sampleNonPartitionList):
                self.distanceBetweenSd()   
            
            if ("concentration" in self.sampleNonPartitionList):
                self.distanceBetweenConcentration()
        return 

    def distanceBetweenPartitions(self):
        partitionDist = partitions.dist_from_clusts(self.XClusts, self.YClusts)
        self.equal["partition"] = (partitionDist == 0)
        self.dists["partition"].append(partitionDist)
        return 

    def distanceBetweenMeans(self, pairings):
        if not self.equal["partition"]:
            self.equal["means"] = False
            self.dists["means"].append(-1) # signify that we don't bother with cluster mean distances when partitions are not equal
        else:
            meanDist = np.zeros((len(self.XClusts), len(self.YClusts)))
            for pair in pairings:
                XInd, YInd = pair
                meanDist[XInd, YInd] = np.linalg.norm(self.XMeans[XInd]-self.YMeans[YInd])
            maxMeanDist = meanDist.max()
            self.equal["means"] = (np.isclose(maxMeanDist,0))
            self.dists["means"].append(maxMeanDist)
        return 

    def distanceBetweenSd(self):
        sdDist = np.linalg.norm(self.Xsd0-self.Ysd0)
        self.equal["sd"] = (np.isclose(sdDist, 0))
        self.dists["sd"].append(sdDist)
        return 

    def distanceBetweenConcentration(self):
        alphaDist = np.abs(self.Xalpha-self.Yalpha)
        self.equal["concentration"] = (np.isclose(alphaDist, 0))
        self.dists["concentration"].append(alphaDist)
        return 

    def checkInvariants(self):
        """
        Runtime:
            O(KM + ND)
        """
        # intersection sizes
        pairwise_dists = partitions.pairwise_dists(self.XClusts, self.YClusts, self.intersectionSizes)
        pairwise_dists_correct = partitions.pairwise_dists(self.XClusts, self.YClusts)
        assert np.alltrue(pairwise_dists==pairwise_dists_correct)

        # clustSums
        XNClust = len(self.XClusts)
        for Xc in range(XNClust):
            XDataIndices = list(self.XClusts[Xc])
            XCorrectSum = np.sum(self.data[XDataIndices,:], axis=0)
            assert np.allclose(XCorrectSum, self.XClustSums[Xc])

        YNClust = len(self.YClusts)
        for Yc in range(YNClust):
            YDataIndices = list(self.YClusts[Yc])
            YCorrectSum = np.sum(self.data[YDataIndices,:], axis=0)
            assert np.allclose(YCorrectSum, self.YClustSums[Yc])
        return 

    # def advanceX(self):
    #     """
    #     Advance the X chain by one step using the marginal distribution.

    #     Runtime:
    #         O(NKD)
    #     """
    #     initZ, _ = partitions.labelHelpersToLabelsAndClusts(self.XLabelList)
    #     initDict = {"sd0": self.Xsd0, "sd1": self.Xsd1, "alpha":self.Xalpha,
    #                 "initZ": initZ, "initType":None}
    #     # sampler = DPMMSingleGibbs(self.data, 
    #     #                                   initDict, 
    #     #                                   self.hyperPriorDict,
    #     #                                   self.fastAssignment,
    #     #                                   self.sampleNonPartitionList,
    #     #                                   np.inf, 
    #     #                                   self.hFunctionDict, 
    #     #                                   self.rng)

    #     sampler = DPMMSingleGibbs(self.data, 
    #                                       initDict, 
    #                                       self.hyperPriorDict,
    #                                       self.fastAssignment,
    #                                       self.sampleNonPartitionList,
    #                                       np.inf, 
    #                                       self.hFunctionDict, 
    #                                       self.rng)
    #     hyperParams, partition = sampler.sampleNSteps(1)

    #     self.Xalpha, self.Xsd0, self.Xsd1 = hyperParams
    #     self.XLabelList, self.XClusts, self.XClustSums, self.XMeans = partition

    #     self.evaluateHFunction(True, False)
    #     return 

    def advanceX(self):
        raise NotImplementedError

    def sampleAssignment(self, n):
        """
        Runtime:
            O(KD + time to solve K x K optimal transport)
        """

        Xc = self.XLabelList[n].getValue()
        Yc = self.YLabelList[n].getValue()
        self.intersectionSizes[Xc, Yc] -= 1

        XRemovedClust = partitions.pop_idx(self.XLabelList, 
                                                            self.XClusts, self.XMeans, 
                                                            self.XClustSums, 
                                                            n, 
                                                            self.data[n,:])
        YRemovedClust = partitions.pop_idx(self.YLabelList, 
                                                            self.YClusts, self.YMeans, 
                                                            self.YClustSums, 
                                                            n, 
                                                            self.data[n,:])

        if XRemovedClust is not None:
            self.intersectionSizes = np.delete(self.intersectionSizes, XRemovedClust, axis=0)
        if YRemovedClust is not None:
            self.intersectionSizes = np.delete(self.intersectionSizes, YRemovedClust, axis=1)

        if ("means" in self.sampleNonPartitionList):
            XThisMeans, YThisMeans = self.XMeans, self.YMeans 
        else:
            XThisMeans, YThisMeans  = None, None

        if (self.fastAssignment):
            XThisClustSums, YThisClustSums = self.XClustSums, self.YClustSums
        else:
            XThisClustSums, YThisClustSums = None, None

        XNewMean, XLocProbs = weights.clusterProbs(self.data, 
                                                    self.XClusts, XThisMeans, XThisClustSums,
                                                    self.Xsd1, self.Xsd0, self.Xalpha, 
                                                    n, self.rng)

        YNewMean, YLocProbs = weights.clusterProbs(self.data, 
                                                    self.YClusts, YThisMeans, YThisClustSums,
                                                    self.Ysd1, self.Ysd0, self.Yalpha, 
                                                    n, self.rng)


        typ, nugget = self.couplingDict["GibbsCoupling"].split("=")
        nugget = float(nugget)
        if typ == "Common_RNG":
            newx, newy = couplings.naive_coupling(XLocProbs, YLocProbs, nugget, self.rng)
        elif typ == "Maximal":
            nugget = float(nugget)
            newx, newy = couplings.max_coupling(XLocProbs, YLocProbs, nugget, self.rng)
        else:
            assert typ == "OT"
            if (self.couplingDict["metric"] == "Hamming"):
                pairwise_dists = partitions.pairwise_dists(self.XClusts, self.YClusts, self.intersectionSizes)
            else:
                assert self.couplingDict["metric"] == "VI"
                pairwise_dists = partitions.pairwise_VI_dists(self.XClusts, self.YClusts, n)
            _, (newx, newy), _ = couplings.optimal_coupling(
                XLocProbs, YLocProbs, pairwise_dists, True, nugget, self.rng)
        
        # if necessary, instantiate a new cluster and pad intersection_sizes appropriately
        if newx == len(self.XClusts):
            self.XClusts.append(set())
            self.XClustSums.append(np.zeros(self.D))
            self.XMeans.append(XNewMean)
            self.intersectionSizes = partitions.pad_with_zeros(self.intersectionSizes, 0)
            XLabelObj = partitions.LabelHelper(newx)
        else:
            XLabelObj = partitions.labelOfCluster(self.XLabelList, self.XClusts, newx)

        if newy == len(self.YClusts):
            self.YClusts.append(set())
            self.YClustSums.append(np.zeros(self.D))
            self.YMeans.append(YNewMean)
            self.intersectionSizes = partitions.pad_with_zeros(self.intersectionSizes, 1)
            YLabelObj = partitions.LabelHelper(newy)
        else:
            YLabelObj = partitions.labelOfCluster(self.YLabelList, self.YClusts, newy)

        # update cluster assigments and intersection sizes
        self.XClusts[newx].add(n); self.YClusts[newy].add(n)
        self.XClustSums[newx] += self.data[n,:]; self.YClustSums[newy] += self.data[n,:]; 
        self.XLabelList[n] = XLabelObj; self.YLabelList[n] = YLabelObj
        self.intersectionSizes[newx,newy] += 1

        return 

    def samplePartition(self):
        raise NotImplementedError

    def sampleMeans(self):

        pairings, XRemClusts, YRemClusts = partitions.find_pairings(self.XClusts, self.YClusts, self.intersectionSizes) 

        for pair in pairings:
            XInd, YInd = pair

            XPosMean, XPosSig = weights.GaussianPosterior(self.data, 
                                                        self.XClusts, self.XClustSums, 
                                                        self.Xsd1, self.Xsd0, XInd)
            YPosMean, YPosSig = weights.GaussianPosterior(self.data, 
                                                        self.YClusts, self.YClustSums, 
                                                        self.Ysd1, self.Ysd0, YInd)
            XMean, YMean, _ = couplings.c_maximal_diag_normal(self.rng, XPosMean, XPosSig, YPosMean, YPosSig)
            self.XMeans[XInd] = XMean
            self.YMeans[YInd] = YMean

        for Xc in XRemClusts:
            XPosMean, XPosSig = weights.GaussianPosterior(self.data, 
                                                        self.XClusts, self.XClustSums,
                                                        self.Xsd1, self.Xsd0, Xc)
            self.XMeans[Xc] = self.rng.normal(loc=XPosMean, scale=np.sqrt(XPosSig))

        for Yc in YRemClusts:
            YPosMean, YPosSig = weights.GaussianPosterior(self.data, 
                                                         self.YClusts, self.YClustSums,
                                                         self.Ysd1, self.Ysd0, Yc)
            self.YMeans[Yc] = self.rng.normal(loc=YPosMean, scale=np.sqrt(YPosSig))

        # check distances
        self.distanceBetweenMeans(pairings)
        return 

    def sampleStandardDeviation(self):

        if (self.hyperPriorDict["separateSd"]):
        
            XPrecShape, XPrecRate = weights.gammaPosteriorPrec(self.data, 
                                                    self.XMeans, self.XLabelList, 
                                                    PRIOR_SD_GAMMA_SHAPE, PRIOR_SD_GAMMA_RATE,
                                                    (1/self.hyperPriorDict["sd1OverSd0"]**2))

            YPrecShape, YPrecRate = weights.gammaPosteriorPrec(self.data, 
                                        self.YMeans, self.YLabelList, 
                                        PRIOR_SD_GAMMA_SHAPE, PRIOR_SD_GAMMA_RATE,
                                        (1/self.hyperPriorDict["sd1OverSd0"]**2))

            XPrec, YPrec = np.zeros(self.D), np.zeros(self.D)

            for d in range(self.D):
                XPrec[d], YPrec[d], _ = couplings.c_maximal_gamma(self.rng,
                                                                    XPrecShape, XPrecRate[d],
                                                                    YPrecShape, YPrecRate[d])

            self.Xsd0, self.Ysd0 = np.sqrt(1/XPrec), np.sqrt(1/YPrec)

        else:
            XPrecShape, XPrecRate = weights.gammaPosteriorPrecIsoSd(self.data, 
                                                    self.XMeans, self.XLabelList, 
                                                    PRIOR_SD_GAMMA_SHAPE, PRIOR_SD_GAMMA_RATE,
                                                    (1/self.hyperPriorDict["sd1OverSd0"]**2))

            YPrecShape, YPrecRate = weights.gammaPosteriorPrecIsoSd(self.data, 
                                        self.YMeans, self.YLabelList, 
                                        PRIOR_SD_GAMMA_SHAPE, PRIOR_SD_GAMMA_RATE,
                                        (1/self.hyperPriorDict["sd1OverSd0"]**2))

            XPrec, YPrec,  _ = couplings.c_maximal_gamma(self.rng,
                                                                    XPrecShape, XPrecRate,
                                                                    YPrecShape, YPrecRate)

            self.Xsd0, self.Ysd0 = np.sqrt(1/XPrec)*np.ones(self.D), np.sqrt(1/YPrec)*np.ones(self.D)

        self.Xsd1, self.Ysd1 = self.hyperPriorDict["sd1OverSd0"]*self.Xsd0, self.hyperPriorDict["sd1OverSd0"]*self.Ysd0

        self.distanceBetweenSd()
        return 

    def sampleConcentration(self):
        """
        Use gamma prior and augmentation scheme as in Escobar and West 1995
        Section 6 to sample DP concentration parameter. 

        Runtime:
            O(ND)
        """

        XEta, YEta, _ = couplings.c_maximal_beta(self.rng, 
                                        self.Xalpha+1, self.N, 
                                        self.Yalpha+1, self.N)

        XNClust = len(self.XClusts); 
        XGammaShape = PRIOR_ALPHA_GAMMA_SHAPE + XNClust - 1
        XGammaRate = PRIOR_ALPHA_GAMMA_RATE - np.log(XEta)
        XRatio = (XGammaShape)/(self.N*XGammaRate)
        XPiEta = XRatio/(XRatio + 1)

        YNClust = len(self.YClusts); 
        YGammaShape = PRIOR_ALPHA_GAMMA_SHAPE + YNClust - 1
        YGammaRate = PRIOR_ALPHA_GAMMA_RATE - np.log(YEta)
        YRatio = (YGammaShape)/(self.N*YGammaRate)
        YPiEta = YRatio/(YRatio + 1)

        XMixt, YMixt, _ = couplings.c_maximal_Bernoulli(self.rng, XPiEta, YPiEta)

        XGammaShape += XMixt; YGammaShape += YMixt; 

        self.Xalpha, self.Yalpha, _ = couplings.c_maximal_gamma(self.rng, 
                                                     XGammaShape, XGammaRate,
                                                     YGammaShape, YGammaRate)

        self.distanceBetweenConcentration()
        return 

    def sampleNSteps(self, nStep):
        for _ in range(nStep):
            self.samplePartition()

            if ("means" in self.sampleNonPartitionList):
                self.sampleMeans()

                if ("sd" in self.sampleNonPartitionList):
                    self.sampleStandardDeviation()

                if ("concentration" in self.sampleNonPartitionList):
                    self.sampleConcentration()

            self.evaluateHFunction(True, True)
            # self.checkInvariants()

        return 

    def areChainsEqual(self):
        return self.equal["partition"] and self.equal["sd"] and self.equal["means"] and self.equal["concentration"]

class DPMMCoupledGibbs(DPMMCoupled):

    def __init__(self, 
                timeBudget, 
                hFunctionDict, 
                rng,
                **kwargs):

        """
        Inputs:
            data: (N,D) array
            initDict, 
            hyperPriorDict, 
            fastAssignment
            sampleNonPartitionList, 
            couplingDict: dict
            timeBudget: scalar
            hFunctionDict: dict:
            rng: BitGenerator
        """

        DPMMCoupled.__init__(self, 
                            timeBudget, 
                            hFunctionDict, 
                            rng,
                            **kwargs)

    def advanceX(self):
        """
        Advance the X chain by one step.

        Runtime:
            O(NKD)
        """
        initZ, _ = partitions.labelHelpersToLabelsAndClusts(self.XLabelList)
        initDict = {"sd0": self.Xsd0, "sd1": self.Xsd1, "alpha":self.Xalpha,
                    "initZ": initZ, "initType":None}
        kwargs = {"initDict": initDict, 
                  "data": self.data, "hyperPriorDict": self.hyperPriorDict,
                  "fastAssignment": self.fastAssignment, "sampleNonPartitionList":self.sampleNonPartitionList}
        sampler = DPMMSingleGibbs(np.inf, 
                                  self.hFunctionDict, 
                                  self.rng,
                                  **kwargs)
        hyperParams, partition = sampler.sampleNSteps(1)

        self.Xalpha, self.Xsd0, self.Xsd1 = hyperParams
        self.XLabelList, self.XClusts, self.XClustSums, self.XMeans = partition

        self.evaluateHFunction(True, False)
        return 

    def samplePartition(self):
        for n in range(self.N):
            self.sampleAssignment(n)

        self.distanceBetweenPartitions()
        return 

class DPMMCoupledSplitMerge(DPMMCoupled):

    def __init__(self, 
                timeBudget, 
                hFunctionDict, 
                rng,
                **kwargs):

        self.splitMergeDict = kwargs["splitMergeDict"]
        DPMMCoupled.__init__(self, 
                            timeBudget, 
                            hFunctionDict, 
                            rng,
                            **kwargs)
        return 

    def advanceX(self):
        initZ, _ = partitions.labelHelpersToLabelsAndClusts(self.XLabelList)
        initDict = {"sd0": self.Xsd0, "sd1": self.Xsd1, "alpha":self.Xalpha,
                    "initZ": initZ, "initType":None}
        kwargs = {"initDict":initDict,
                  "data": self.data, "hyperPriorDict": self.hyperPriorDict,
                  "splitMergeDict":self.splitMergeDict,
                  "fastAssignment": self.fastAssignment, "sampleNonPartitionList":self.sampleNonPartitionList}
        sampler = DPMMSingleSplitMerge(np.inf, 
                                  self.hFunctionDict, 
                                  self.rng,
                                  **kwargs)
        hyperParams, partition = sampler.sampleNSteps(1)

        self.Xalpha, self.Xsd0, self.Xsd1 = hyperParams
        self.XLabelList, self.XClusts, self.XClustSums, self.XMeans = partition

        self.evaluateHFunction(True, False)

    def samplePartition(self):
        # most naive coupling of the split-merge proposals: use the same
        # i, j but don't actively couple the launch state, the acceptance probability etc
        for _ in range(self.splitMergeDict["numProp"]):
            i, j = self.rng.choice(self.N, 2, replace=False)
            XSmObj = weights.SplitMergeMove(self.data, self.Xsd1, self.Xsd0, self.Xalpha,
                                         self.splitMergeDict, 
                                         self.XLabelList, self.XClusts,
                                         self.rng)
            YSmObj = weights.SplitMergeMove(self.data, self.Ysd1, self.Ysd0, self.Yalpha,
                                         self.splitMergeDict, 
                                         self.YLabelList, self.YClusts,
                                         self.rng)

            Xaccept, self.XLabelList, self.XClusts = XSmObj.sample(i, j)
            Yaccept, self.YLabelList, self.YClusts = YSmObj.sample(i, j)
            self.computeAuxiliary()

        # use DPMMCoupled for the Gibbs steps following that split merge
        # first, compute the intersection sizes since split merge step
        # doesn't update this
        self.setIntersectionSizes()
        for _ in range(self.splitMergeDict["numScan"]):
            for n in range(self.N):
                self.sampleAssignment(n)

        self.distanceBetweenPartitions()
        return

class GraphCoupled(MCMCCoupled):
    def __init__(self, 
                timeBudget, 
                hFunctionDict, 
                rng,
                **kwargs):

        MCMCCoupled.__init__(self,
                            timeBudget, 
                            hFunctionDict, 
                            rng,
                            **kwargs)

        self.graph = kwargs["graph"]
        self.nColors = kwargs["initDict"]["nColors"]
        self.N = self.graph.vcount()

        self.equal = {"partition": False}
        self.dists = {"partition": []}

        self.remainingTime = timeBudget
        self.setUpTime = self.setUp(kwargs["initDict"])
        self.remainingTime = self.remainingTime - self.setUpTime
        self.estimateTime = 0
        return 

    def advanceX(self):
        """
        Advance the X chain by one step.

        Runtime:
        """
        initZ, _ = partitions.labelHelpersToLabelsAndClusts(self.XLabelList)
        initDict = {"initZ": initZ, "initType":None, "nColors": self.nColors}
        kwargs = {"initDict": initDict, "graph": self.graph}
        sampler = GraphSingle(np.inf,   
                                          self.hFunctionDict, 
                                          self.rng, **kwargs)
        self.XLabelList, self.XClusts = sampler.sampleNSteps(1)
        self.evaluateHFunction(True, False)
        return 

    def distanceBetweenPartitions(self):
        partitionDist = partitions.dist_from_clusts(self.XClusts, self.YClusts)
        self.equal["partition"] = (partitionDist == 0)
        self.dists["partition"].append(partitionDist)
        return 

    def distanceBetweenChains(self):
        self.distanceBetweenPartitions()
        return

    def areChainsEqual(self):
        return self.equal["partition"]

    def setInitialState(self, initDict):
        assert initDict["initType"] == "greedy"
        _, self.XLabelList, self.XClusts = priors.greedyColoring(self.graph, self.nColors)
        _, self.YLabelList, self.YClusts = priors.greedyColoring(self.graph, self.nColors)
        # function values along Markov chain
        self.hX, self.hY = [], []
        self.evaluateHFunction(True, True)
        return 

    def sampleAssignment(self, n):

        Xc = self.XLabelList[n].getValue()
        Yc = self.YLabelList[n].getValue()
        self.intersectionSizes[Xc, Yc] -= 1
        
        XRemovedClust = partitions.pop_idx(self.XLabelList, 
                                                            self.XClusts, None, 
                                                            None, 
                                                            n, 
                                                            None)
        
        YRemovedClust = partitions.pop_idx(self.YLabelList, 
                                                            self.YClusts, None, 
                                                            None, 
                                                            n, 
                                                            None)

        if XRemovedClust is not None:
            self.intersectionSizes = np.delete(self.intersectionSizes, XRemovedClust, axis=0)
        if YRemovedClust is not None:
            self.intersectionSizes = np.delete(self.intersectionSizes, YRemovedClust, axis=1)

        XLocProbs = weights.colorProbs(self.graph, self.nColors, n, 
                                   self.XLabelList, self.XClusts)
        YLocProbs = weights.colorProbs(self.graph, self.nColors, n, 
                                   self.YLabelList, self.YClusts)

        typ, nugget = self.couplingDict["GibbsCoupling"].split("=")
        nugget = float(nugget)
        if typ == "Common_RNG":
            newx, newy = couplings.naive_coupling(XLocProbs, YLocProbs, nugget, self.rng)
        elif typ == "Maximal":
            nugget = float(nugget)
            newx, newy = couplings.max_coupling(XLocProbs, YLocProbs, nugget, self.rng)
        else:
            assert typ == "OT"
            # allow new clusters iff current coloring uses less than nColors colors
            allowNewClustX = (len(self.XClusts) < self.nColors)
            allowNewClustY = (len(self.YClusts) < self.nColors)
            if (self.couplingDict["metric"] == "Hamming"):
                pairwise_dists = partitions.pairwise_dists(self.XClusts, self.YClusts, 
                                                          self.intersectionSizes, allowNewClustX, allowNewClustY)
            else:
                assert self.couplingDict["metric"] == "VI"
                pairwise_dists = partitions.pairwise_VI_dists(self.XClusts, self.YClusts, n,
                                                            allowNewClustX, allowNewClustY)
            # print("n = %d" %n)
            # print("self.XClusts", self.XClusts)
            # print("self.YClusts", self.YClusts)
            # print("XLocProbs", XLocProbs)
            # print("YLocProbs", YLocProbs)
            # print("pairwise_dists.shape", pairwise_dists.shape)
            _, (newx, newy), _ = couplings.optimal_coupling(
                XLocProbs, YLocProbs, pairwise_dists, True, nugget, self.rng)
        
        # if necessary, instantiate a new cluster and pad intersection_sizes appropriately
        if newx == len(self.XClusts):
            self.XClusts.append(set())
            self.intersectionSizes = partitions.pad_with_zeros(self.intersectionSizes, 0)
            XLabelObj = partitions.LabelHelper(newx)
        else:
            XLabelObj = partitions.labelOfCluster(self.XLabelList, self.XClusts, newx)

        if newy == len(self.YClusts):
            self.YClusts.append(set())
            self.intersectionSizes = partitions.pad_with_zeros(self.intersectionSizes, 1)
            YLabelObj = partitions.LabelHelper(newy)
        else:
            YLabelObj = partitions.labelOfCluster(self.YLabelList, self.YClusts, newy)

        # update cluster assigments and intersection sizes
        self.XClusts[newx].add(n); self.YClusts[newy].add(n)
        self.XLabelList[n] = XLabelObj; self.YLabelList[n] = YLabelObj
        self.intersectionSizes[newx,newy] += 1

        return 

    def checkInvariants(self):
        # intersection sizes
        pairwise_dists = partitions.pairwise_dists(self.XClusts, self.YClusts, self.intersectionSizes)
        pairwise_dists_correct = partitions.pairwise_dists(self.XClusts, self.YClusts)
        assert np.alltrue(pairwise_dists==pairwise_dists_correct)

        # number of clusters is less than number of colors
        assert len(self.XClusts) <= self.nColors and len(self.YClusts) <= self.nColors
        
        # check that self.clusts[i] are all the data points n where self.labelList[n].getValue() 
        # is equal to i.
        XNClust = len(self.XClusts)
        for i in range(XNClust):
            dataIndices = self.XClusts[i]
            values = [self.XLabelList[n].getValue() for n in dataIndices]
            assert np.alltrue(values == i*np.ones(len(dataIndices)))
            
        YNClust = len(self.YClusts)
        for i in range(YNClust):
            dataIndices = self.YClusts[i]
            values = [self.YLabelList[n].getValue() for n in dataIndices]
            assert np.alltrue(values == i*np.ones(len(dataIndices)))
        
        return 

    def samplePartition(self):
        for n in range(self.N):
            self.sampleAssignment(n)
            if (DEBUG):
                self.checkInvariants()

        self.distanceBetweenPartitions()
        return

    def sampleNSteps(self, nStep):
        for _ in range(nStep):
            self.samplePartition()
            self.evaluateHFunction(True, True)
            if (DEBUG):
                self.checkInvariants()
        return 

def extractUnbiasedEstimate(M, burnIn, minIter, hX, hY, tau):
    """computes an unbiased estimate from two coupled chains following
    Jacob 2020 equation 2.1

    Inputs:
        M: scalar, number of output dimensions of function
        burnIn: scalar
        minIter: scalar
        hX: (tX,), list
        hY: (tY,) list
        tau: scalar, meeting time, measured in X chain's time

    Return:

    Runtime:

    """
    assert minIter > burnIn

    if tau == -1:
        return [math.nan for _ in range(M)]

    assert len(hY) == tau
    if tau <= minIter:
        assert len(hX) == minIter + 1
    else:
        assert len(hX) == tau + 1

    # compute first term (usual MCMC estimate)
    term1 = np.mean(np.array(hX[burnIn:minIter+1]), axis=0) # h(X_burnIn) to h(X_minIter)

    # compute second term (bias correction)
    ls = np.arange(burnIn+1, tau)
    term2_scalings = np.array([min([1, (l-burnIn)/(minIter-burnIn+1)]) for l in ls])
    term2_diffs = np.array([hX[l] - hY[l-1] for l in ls])
    term2 = np.tensordot(term2_scalings, term2_diffs, axes=[[0],[0]])

    return term1 + term2