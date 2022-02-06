"""
Read from command line using names_and_parser, define functions 
for parallel processing and run them. 


SHORTERM TODO:
- recover all the different data options with dataUtils
- incorporate the fastAssignment option
- incorporate control flow to use DPMMSingleSplitMerge as well as DPMMSingleGibbs
    - perhaps it's easier to directly specify which functions from the sampler_* modules to 
        use for estimation. it'll simplify the comparison between mixture and graph, also
        between Gibbs and splitMerge
- use the json functionality to overwrite some command line arguments (storting 
experiment settings as json is much more readable)
- remove references to the lastIterate field of estInfo dicts
- unify the various data handling files

LONGTERM TODO:
- unify make_gmm and load_gmm dataRoot options
"""

# Standard libaries
import os, sys
import json
import numpy as np
np.set_printoptions(precision=4)
import time
import pandas as pd
from numpy.random import SeedSequence, default_rng

# our implementation
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import interesting_functions
import samplerLib
from dataUtils import loadMixtureData, loadGraphData
# import gene_data, graph_data, gmm_data, UCIDataSets
# import gmm_data
import names_and_parser

NREP_TRACE_SAVE_LIMIT=5
DISTS_SAVE_LIMIT=600

## ------------------------------------------------------------------

options = names_and_parser.parse_args()
optionsAsDict = vars(options) # alternate view of options

with open(options.configPath) as f:
    config = json.load(f)
optionsAsDict.update(config)

# All config
print('===========================')
for key, val in vars(options).items():
    print('{}: {}'.format(key, val))
print('===========================')

dirName, estFileName, auxFileName = names_and_parser.makeDirAndFileName(options)
print("Savedir is %s" %dirName)

# seed sequence for parallel but reproducible computing.
ss = SeedSequence(12345)
childrenSeeds = ss.spawn(2*options.nRep)
streams = [default_rng(s) for s in childrenSeeds]

if os.path.exists(dirName):
    listDir = os.listdir(dirName)
    if (len(listDir) > 0) and (not options.overwrite):
        print("Results exist, skipping since overwrite = False")
        sys.exit(0)
else:
    print("Will make directory %s" %dirName)
    os.makedirs(dirName)
    if (options.justMakeDir):
        print("Just making dir, not running anything.")
        sys.exit(0)

# modifiers of various sampling classes
kwargs = {}

if (options.samplerArgs["estType"] == "coupled"):
    maxTimeFn = lambda seed: options.samplerArgs["maxTime"]
        
    couplingArgs = options.samplerArgs["couplingArgs"]

    kwargs["couplingDict"] = {"GibbsCoupling": couplingArgs["GibbsCoupling"], 
                              "metric": couplingArgs["metric"]}

    thisBurnIn, thisNIter = couplingArgs["burnIn"], couplingArgs["minIter"]
else:
    if not options.samplerArgs["singleArgs"]["loadTime"]:
        maxTimeFn = lambda seed: options.samplerArgs["maxTime"]
    else:
        # load time taken from coupled results
        assert options.samplerArgs["estType"] == "single"
        couplingArgs = options.samplerArgs["couplingArgs"]
        assert couplingArgs["GibbsCoupling"] == "OT=1e-5" and couplingArgs["metric"] == "Hamming"
        
        # switch some arguments to load 
        options.samplerArgs["estType"] = "coupled"
        oldName = options.samplerName
        options.samplerName = couplingArgs["samplerName"]
        _, otherFileName, _ = names_and_parser.makeDirAndFileName(options)
        # reset arguments after load
        options.samplerArgs["estType"] = "single"
        options.samplerName = oldName
        
        coupledResults = pd.read_csv(otherFileName, delimiter=",")
        maxTimeFn = lambda i: coupledResults[coupledResults["seed"]==i]["timeTaken"].iloc[0]
        print("Total time in seconds for computing as recorded by coupled estimates is %.2f" \
              %np.sum(coupledResults["timeTaken"]))

    maxIter = options.samplerArgs["singleArgs"]["maxIter"]
    if (maxIter == -1):
        # run until time runs out
        thisBurnIn, thisNIter = -1, -1
    else:
        # run until gven number of steps
        thisBurnIn, thisNIter = int(maxIter*samplerLib.DEFAULT_BURNIN_RATIO), maxIter


## ------------------------------------------------------------------
## data, probabilistic model, other sampler modifiers
if (options.modelType == "mixture"):
    data = loadMixtureData(options.dataDir, options.dataName)
    nObs, D = data.shape
    kwargs["data"] = data

    # initialization 
    sd0_arr = options.modelArgs["sd0"]*np.ones(D)
    initDict = {"sd1": np.full(D, options.modelArgs["sd1"]), 
                "sd0": np.full(D, options.modelArgs["sd0"]), 
                "alpha":options.modelArgs["alpha"], 

                "initZ": None, "initType": options.samplerArgs["initType"]}
    # what to sample
    sampleNonPartitionList = names_and_parser.strToList(options.modelArgs["sampleNonPartitionStr"])
    # hyper priors
    hyperPriorDict = {"sd1OverSd0": options.modelArgs["sd1OverSd0"], "separateSd": options.modelArgs["separateSd"]}
    kwargs.update({"initDict":initDict, "sampleNonPartitionList": sampleNonPartitionList, "hyperPriorDict":hyperPriorDict})
    
    # split merge specifics
    if "splitMergeDict" in options.samplerArgs:
        kwargs.update({"splitMergeDict":options.samplerArgs["splitMergeDict"]})
    # do we use fast assignment
    kwargs.update({"fastAssignment": options.samplerArgs["fastAssignment"]})

else:
    assert options.modelType == "graphColoring"
    graph = loadGraphData(options.dataDir, options.dataName)
    nObs = graph.vcount()
    kwargs["graph"] = graph

    initDict = {"initZ": None, "initType": options.samplerArgs["initType"],
                "nColors": options.modelArgs["nColors"]}
    kwargs.update({"initDict":initDict})
    
## ------------------------------------------------------------------
# integrands of interest
if (options.funcArgs["hType"] == "LCP"):
    topK = 10
    hFunction = lambda labelList, clusts: interesting_functions.LCP(clusts, k=topK)
    nHColumns = topK
elif (options.funcArgs["hType"] == "CC"):
    if (options.samplerArgs["estType"] == "truth"):
        CCIndices = range(0, nObs) # for ground truth, estimate adjacency average for all data points
    else:
        CCIndices = options.funcArgs["CCIndices"] # for estimates, estimate only a submatrix for memory reasons
    hFunction = lambda labelList, clusts: interesting_functions.CC(labelList, CCIndices)
    nHColumns = np.square(len(CCIndices))
elif (options.funcArgs["hType"] == "ppd"):
    grid = grid_and_density[0]
    hFunction = lambda labelList, clusts: interesting_functions.posterior_predictive_density_1Dgrid(grid, data, clusts, sd1, sd0, alpha)
    nHColumns = len(grid)
hFunctionDict = {"hFunction": hFunction, "nHColumns": nHColumns}

samplerClass = getattr(samplerLib, options.samplerName)
print(samplerClass)

## ------------------------------------------------------------------
# done with set up, now experiment! 

def rep(i):
    rng = streams[i]
    timeBudget = np.inf if maxTimeFn(i) == -1 else maxTimeFn(i)
    sampler = samplerClass(timeBudget, hFunctionDict, rng, **kwargs)
    estInfo, timeInfo, auxInfo = sampler.estimate(thisBurnIn, thisNIter)
    timeTaken = sampler.getTimeTaken()
    return estInfo, timeInfo, auxInfo, timeTaken

allSeeds = range(options.nRep)
thisSeeds = allSeeds[options.taskID:options.nRep:options.nTasks]

print("Total number of replicates to do is %d" %len(thisSeeds))
progress_count = max(1, int(len(thisSeeds)/10))

st = time.time()

file = open(estFileName, "w")
headerLine = names_and_parser.makeHeaderLine(nHColumns)
file.write(headerLine + "\n")

auxFile = open(auxFileName, "w")
auxInfoList = []

for idx, seed in enumerate(thisSeeds):
    estInfo, timeInfo, auxInfo, timeTaken = rep(seed)
    line = names_and_parser.resultsToLine(seed, 
                                          estInfo["timeAverage"], 
                                          timeInfo["hasMet"], 
                                          timeInfo["tau"], 
                                          timeInfo["nStep"], 
                                          timeInfo["nXstep"],
                                          timeTaken)
    file.write(line + "\n")
    file.flush() # write results as they come
    if (idx + 1) % progress_count == 0:
        print("Finished another %d replicates.\n" %progress_count)
        
    if (len(allSeeds) <= DISTS_SAVE_LIMIT):
        auxInfo.update({"seed": seed})
        auxInfoList.append(auxInfo)
        json.dump(auxInfoList, auxFile, indent=4)

auxFile.close()
file.close()
    
print("Wall-clock time time elasped in minutes %.2f" %((time.time()-st)/60))
print("done ------------------------------------ \n")