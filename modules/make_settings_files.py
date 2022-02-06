import numpy as np
import pandas as pd
import itertools

from copy import deepcopy

import json


def makeKRegular():
    dict_ = {}

    dict_.update({"dataName": "kRegular.pkl"})

    dict_.update({"modelType": "graphColoring"})

    modelArgs = {"nColors": 4}
    dict_.update({"modelArgs": modelArgs})

    dict_.update({"resultsDir": "../results/"})

    funcArgs = {"hType": "CC", "CCIndices": [0, 1, 2, 3, 4, 5]}
    dict_.update({"funcArgs": funcArgs})

    dict_.update({"samplerName": "GraphCoupled"})
    couplingArgs = {"burnIn":1, "minIter":4,
                    "GibbsCoupling": "OT=1e-5", "metric": "Hamming"}
    samplerArgs = { "estType":"coupled",
                    "couplingArgs": couplingArgs,
                    "initType": "greedy",
                    "maxTime": -1}
    dict_.update({"samplerArgs": samplerArgs})

    path = "../examples/graphColoring/kRegular.json"
    json.dump(dict_, open(path, "w"), indent=4)

    return 

def makeGeneGibbsFixedDP():
    dict_ = {}

    dict_.update({"dataName": "gene.csv"})

    dict_.update({"modelType": "mixture"})
    modelArgs = {"sd0": 0.5, "sd1": 1.3, "alpha":1.0,
                "sampleNonPartitionStr":"",
                "sd1OverSd0":-1, "separateSd": False}
    dict_.update({"modelArgs": modelArgs})
    
    funcArgs = {"hType": "LCP"}
    dict_.update({"funcArgs": funcArgs})

    thisCopy = deepcopy(dict_)

    thisCopy.update({"samplerName": "DPMMCoupledGibbs"})
    couplingArgs = {"burnIn":10, "minIter":50,
                    "GibbsCoupling": "OT=1e-5", "metric": "Hamming"}
    samplerArgs = { "estType":"coupled",
                    "couplingArgs": couplingArgs,
                    "initType": "allSame",
                    "maxTime": -1, 
                    "fastAssignment": False}
    thisCopy.update({"samplerArgs": samplerArgs})

    path = "../submissions/aistats2022CameraReady/gene.csv/coupledConfig.json"
    json.dump(thisCopy, open(path, "w"), indent=4)

    # ground truth
    thisCopy = deepcopy(dict_)

    thisCopy.update({"nRep":10})

    thisCopy.update({"samplerName": "DPMMSingleGibbs"})
    singleArgs = {"maxIter": 10000, "loadTime": False}
    samplerArgs = { "estType":"truth",
                    "singleArgs": singleArgs,
                    "initType": "allSame",
                    "maxTime": -1, 
                    "fastAssignment": False}
    thisCopy.update({"samplerArgs": samplerArgs})

    path = "../submissions/aistats2022CameraReady/gene.csv/groundTruthConfig.json"
    json.dump(thisCopy, open(path, "w"), indent=4)
    return 

def makeSynthetcicN500GibbsVariedDP():
    dict_ = {}

    dict_.update({"dataName": "synthetic_N=500.csv"})

    dict_.update({"modelType": "mixture"})
    modelArgs = {"sd0": 2.0, "sd1": 0.5, "alpha":0.1,
                "sampleNonPartitionStr":"means,sd,concentration",
                "sd1OverSd0":0.7, "separateSd": True}
    dict_.update({"modelArgs": modelArgs})
    
    dict_.update({"resultsDir": "../results/"})

    funcArgs = {"hType": "LCP"}
    dict_.update({"funcArgs": funcArgs})

    dict_.update({"samplerName": "DPMMCoupledGibbs"})
    couplingArgs = {"burnIn":5, "minIter":10,
                    "GibbsCoupling": "OT=1e-5", "metric": "Hamming"}
    samplerArgs = { "estType":"coupled",
                    "couplingArgs": couplingArgs,
                    "initType": "allSame",
                    "maxTime": -1, 
                    "fastAssignment": False}
    dict_.update({"samplerArgs": samplerArgs})

    path = "../examples/mixtureModel/Gibbs/synthetic_N=500_GibbsVariedDP.json"
    json.dump(dict_, open(path, "w"), indent=4)

    return 

def makeSynthetcicN500Other():
    dict_ = {}

    dict_.update({"dataName": "synthetic_N=500.csv"})

    dict_.update({"modelType": "mixture"})
    modelArgs = {"sd0": 2.0, "sd1": 0.5, "alpha":0.1,
                "sampleNonPartitionStr":"means,sd,concentration",
                "sd1OverSd0":0.7, "separateSd": True}
    dict_.update({"modelArgs": modelArgs})
    
    dict_.update({"resultsDir": "../results/"})

    funcArgs = {"hType": "LCP"}
    dict_.update({"funcArgs": funcArgs})

    # single
    thisCopy = deepcopy(dict_)
    thisCopy.update({"samplerName": "DPMMSingleGibbs"})
    couplingArgs = {"burnIn":5, "minIter":10,
                    "GibbsCoupling": "OT=1e-5", "metric": "Hamming"}
    singleArgs = {"loadTime": True, "maxIter": -1}
    samplerArgs = { "estType":"single",
                    "singleArgs": singleArgs,
                    "couplingArgs": couplingArgs,
                    "initType": "allSame",
                    "maxTime": -1, 
                    "fastAssignment": False}
    thisCopy.update({"samplerArgs": samplerArgs})
    path = "../examples/mixtureModel/Gibbs/synthetic_N=500_single.json"
    json.dump(thisCopy, open(path, "w"), indent=4)

    # ground truth
    thisCopy = deepcopy(dict_)
    thisCopy.update({"samplerName": "DPMMSingleGibbs"})
    singleArgs = {"loadTime": False, "maxIter": 10000}
    samplerArgs = { "estType":"truth",
                    "singleArgs": singleArgs,
                    "initType": "allSame",
                    "maxTime": -1, 
                    "fastAssignment": False}
    thisCopy.update({"samplerArgs": samplerArgs})
    path = "../examples/mixtureModel/Gibbs/synthetic_N=500_truth.json"
    json.dump(thisCopy, open(path, "w"), indent=4)

    # coupling with other coupling mechanisms
    thisCopy = deepcopy(dict_)
    thisCopy.update({"samplerName": "DPMMCoupledGibbs"})
    couplingArgs = {"burnIn":5, "minIter":10,
                    "GibbsCoupling": "Common_RNG=1e-5"}
    singleArgs = {"loadTime": True, "maxIter": -1}
    samplerArgs = { "estType":"coupled",
                    "couplingArgs": couplingArgs,
                    "initType": "allSame",
                    "maxTime": -1, 
                    "fastAssignment": False}
    thisCopy.update({"samplerArgs": samplerArgs})
    path = "../examples/mixtureModel/Gibbs/synthetic_N=500_commonRng.json"
    json.dump(thisCopy, open(path, "w"), indent=4)

    # coupling with other coupling mechanisms
    thisCopy = deepcopy(dict_)
    thisCopy.update({"samplerName": "DPMMCoupledGibbs"})
    couplingArgs = {"burnIn":5, "minIter":10,
                    "GibbsCoupling": "Maximal=1e-5"}
    singleArgs = {"loadTime": True, "maxIter": -1}
    samplerArgs = { "estType":"coupled",
                    "couplingArgs": couplingArgs,
                    "initType": "allSame",
                    "maxTime": -1, 
                    "fastAssignment": False}
    thisCopy.update({"samplerArgs": samplerArgs})
    path = "../examples/mixtureModel/Gibbs/synthetic_N=500_Maximal.json"
    json.dump(thisCopy, open(path, "w"), indent=4)

    return 

def makeGeneSplitMergeFixedDP():
    dict_ = {}

    dict_.update({"dataName": "gene.csv"})

    dict_.update({"modelType": "mixture"})
    modelArgs = {"sd0": 0.5, "sd1": 1.3, "alpha":1.0,
                "sampleNonPartitionStr":"",
                "sd1OverSd0":-1, "separateSd": False}
    dict_.update({"modelArgs": modelArgs})
    
    dict_.update({"resultsDir": "../results/"})

    funcArgs = {"hType": "LCP"}
    dict_.update({"funcArgs": funcArgs})

    dict_.update({"samplerName": "DPMMCoupledSplitMerge"})
    couplingArgs = {"burnIn":10, "minIter":50,
                    "GibbsCoupling": "OT=1e-5", "metric": "Hamming"}
    samplerArgs = { "estType":"coupled",
                    "couplingArgs": couplingArgs,
                    "initType": "allSame",
                    "maxTime": -1, 
                    "splitMergeDict":{"numRes":5, "numProp": 1, "numScan":1},
                    "fastAssignment": False}
    dict_.update({"samplerArgs": samplerArgs})

    path = "../examples/mixtureModel/Gibbs/geneSplitMergeFixedDP.json"
    json.dump(dict_, open(path, "w"), indent=4)

    return

def silly():
    sd0 = np.arange(start=1.0, stop=2.1, step=0.25)
    sd = np.arange(start=0.5,stop=1.05,step=0.1)
    
    sd0_w_sd = np.array([x for x in itertools.product(sd0, sd)])
    df = pd.DataFrame(sd0_w_sd, columns=["sd0","sd"])
    df = df.round(2)
    df.to_csv("sd_sd0_alpha.csv")
    return 

if __name__ == "__main__":
    makeGeneGibbsFixedDP()

    # makeGeneSplitMergeFixedDP()

    # makeKRegular()

    # makeSynthetcicN500GibbsVariedDP()

    # makeSynthetcicN500Other()
    