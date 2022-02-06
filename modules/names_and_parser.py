"""
Define functions that read in command line arguments and functions
that return names of directories.





SHORTERM TODO:
    The new structure of the names_and_parser.py is inspired by pytorch-sso/examples/classification/main.py. 

    group the commmand line inputs into smaller dictionaries. For instance, the modelArgs 
    for mixture is generally different than for graphColoring.
        It saves the step in experiment.py where we have to combine various 
            keys in options to make directories.

    group the settings for compilation into a json dictionary as well.
        We don't need the compiling settings for running the experiment.

    need to change the signature of the functions to make the save directory.

    use the kwargs to put the split-merge relevant information into separate json
    rather than 

LONGTERM TODO:
    remove the savepath output from make_est_template since we only 
    save the ests these days.

    The hType field appears in both make_dirName and make_est_template;
    should make it appear in only one place.
"""

import pandas as pd
import argparse
import os 
import re
import fnmatch
import math
import json

## our implementation 
# import gene_data
# import gmm_data

## ------------------------------------------------------------------
## Argument parsers

def parse_args():
    parser = argparse.ArgumentParser()

    ## newest arguments
    
    ## config json file to simplify command line argument
    parser.add_argument("--configPath", type=str, dest="configPath", default="",
                        help='config json file to simplify command line argument')

    ## information about data
    parser.add_argument("--dataName", type=str, dest="dataName", default="gene.csv",
                    help="type of data (synthetic or gene expression data)")
    # parser.add_argument("--dataSuffix", type=str, dest="dataSuffix", default="",
    #                 help="identifying information")
    parser.add_argument("--dataDir", type=str, dest="dataDir", default="../data/",
                    help="root directory containing data files")

    ## save directory
    parser.add_argument("--resultsDir", type=str, dest="resultsDir", default="../results",
                    help="root directory to save results. Will add model args, sampler args etc. ")
    parser.add_argument("--shortDir", type=str, dest="shortDir", default="../temp",
                    help="short directory to store results, doesn't contain model args, sampler args etc.")

    ## information about probablistic model
    parser.add_argument("--modelType", type=str, dest="modelType", default="mixture",
                            choices=["mixture", "graphColoring"], help='kind of model')
    parser.add_argument("--modelArgs", type=json.loads, dest="modelArgs", default=dict(),
                    help="model hyper-parameters")

    # parser.add_argument("--sd1", type=float, dest="sd1", default=5.0,
    #                 help="std of observational likelihood")
    # parser.add_argument("--sd0", type=float, dest="sd0", default=5.0, 
    #                 help="std of prior distribution over cluster means")
    # parser.add_argument("--alpha", type=float, dest="alpha", default=1.0,
    #                 help="concentration of Dirichlet parameter to generate cluster weights")
    # parser.add_argument("--sampleNonPartitionStr", type=str, dest="sampleNonPartitionStr", default="",
    #                 help="types of DP hyper-params to be sampled")
    # parser.add_argument("--sd1OverSd0", type=float, dest="sd1OverSd0", default=-1,
    #                 help="ratio between sd1 and sd0 in case sds are sampled")
    # parser.add_argument("--separateSd", action='store_true', dest="separateSd",
    #                 help="do we have separate sd for separate dimensions")
    # graph coloring stuff
    # parser.add_argument("--nColors", type=int, dest="nColors", default=4,
    #                 help="number of colors to color graph")
    
    ## settings for estimator construction
    
    ## overwrite
    parser.add_argument("--overwrite", dest="overwrite", action="store_true",
                        help="do we overwrite existing results (only at make_dirName() level")
    parser.add_argument("--justMakeDir", dest="justMakeDir", action="store_true",
                        help="do we run experiments or just make directory (avoid conflict in tripleMode")

    ## options for compilation, not necessary for generating the results, per se.
    parser.add_argument("--compileArgs", type=json.loads, dest="compileArgs", default=dict(),
                    help="functions arguments")
    # parser.add_argument("--leftTrimPer", type=float, dest="leftTrimPer", default=0.005)
    # parser.add_argument("--rightTrimPer", type=float, dest="rightTrimPer", default=0.005)
    # parser.add_argument("--compileType", type=str, dest="compileType",
    #                 help="type of compilation to do")
    # parser.add_argument("--compile_settings_file", type=str, dest="compile_settings_file",
    #                 help="file with settings for making comparison")
    #  parser.add_argument("--verbose", dest="verbose", action='store_true',
    #                 help='if false, plot only most relevant figure')

    ## information about function of interest
    parser.add_argument("--funcArgs", type=json.loads, dest="funcArgs", default=dict(),
                    help="functions arguments")

    # parser.add_argument("--hType", type=str, dest="hType", default=["LCP", "CC"], 
    #                         help="type of function")
    # parser.add_argument("--CCIndicesFile",  type=str, dest="CCIndicesFile", default="", 
    #                         help="path to list of indices to compute co-clustering probabiltiy")
    # parser.add_argument("--topK_idx", type=int, dest="topK_idx", default=0,
    #                 help="rank of component proportions (largest, second largest etc)")
    # parser.add_argument("--co_cluster_pair", type=str, dest="co_cluster_pair", default="0,1",
    #                 help="to index into co_clustering matrix (not data index, since co_clustering matrix compares only subset")
   
    ## time budget 
    # parser.add_argument("--loadTime", action="store_true", dest="loadTime", 
    #                     help="whether to use compute time from coupling experiment to set own maxTime")
    # parser.add_argument("--maxIter", type=int, dest="maxIter", default=-1, 
    #                     help="maximum number of sweeps through data when computing truth")
    # parser.add_argument("--maxTime", type=float, dest="maxTime", default=-1.0,
    #                     help="maximum processor time in seconds to run each replicate. If maxTime=-1, no limit on runtime.")
    
    ## and how to run replicates (number of replicates, whether use triple mode) 
    parser.add_argument("--nRep", dest="nRep", type=int, default=1,
                        help='total number of replicates for tripleMode, typically really large')
    parser.add_argument("--taskID", dest="taskID",type=int, default=0,
                        help="node ID for LLSub triples mode")
    parser.add_argument("--nTasks", dest="nTasks",type=int, default=1,
                        help="number of processes for LLSub triples mode")
    
    ## settings for estimator construction


    # parser.add_argument("--estType", type=str, dest="estType", default="coupled",
    #                 help="type of estimator", choices=["truth", "coupled", "single"])
    # parser.add_argument("--burnIn", type=int, dest="burnIn", default=1,
    #                          help="length of burn-in period. \
    #                     Convention: if burnIn = -1, we actually discard the first 10 percent of sweeps.")
    # parser.add_argument("--minIter", type=int, dest="minIter",default=-1,
    #                 help="minimum number of sweeps to perform when coupling")
    # parser.add_argument("--initType", type=str, dest="initType", default="allSame",
    #                 help="how to initialize the Gibbs sampler")
    
    ## settings for split-merge, relevant to sampler settings
    parser.add_argument("--samplerName", dest="samplerName", type=str,
                        default='DPMMGibbsSingle', help='')
    parser.add_argument("--samplerArgs", type=json.loads, dest="samplerArgs", default=dict(),
                    help="sampler settings")
    # parser.add_argument("--samplerType", type=str, dest="samplerType", default="Gibbs", 
    #                         choices=["Gibbs", "JainNeal"])
    # parser.add_argument("--GibbsCoupling",type=str, dest='GibbsCoupling',
    #                     help="how to couple individual label re-assignment.")
    # parser.add_argument("--metric", type=str, help="choice of metric between partitions", default="Hamming",
    #                     dest='metric')
    # parser.add_argument("--fastAssignment", dest="fastAssignment", action='store_true',
    #                     help='')
    # parser.add_argument("--num_prop", type=int, dest="num_prop", default=1,
    #                     help="number of split-merge proposals during each Jain-Neal iteration")
    # parser.add_argument("--t", type=int, dest="t", default=5,
    #                     help="number of restricted Gibbs scan to reach launch state")
    # parser.add_argument("--num_scans", type=int, dest="num_scans", default=1,
    #                     help="number of Gibbs scan after all split-merge proposals")
   
    parser.add_argument("--profile",  dest="profile", 
                        help="whether to profile the functions", action='store_true')
    # parser.add_argument("--debug",  dest="debug", 
    #                     help="if debug, don't use parallel jobs", action='store_true')

    options = parser.parse_args()
    return options

def parse_compile_file(path):
    """
    Read type I of compilation settings 
        first line is vary_key name 
        second line gives the set of vary_items. 
    """
    file = open(path,"r")
    dict_ = {}
    lines = file.readlines()
    for (idx, line) in enumerate(lines):
        line = line.replace("\n","")
        blocks = line.split(",")
        key = blocks[0]
        if (key == "vary_key"):
            val_as_str = blocks[1]
            val = val_as_str
        elif (key == "var_items"):
            data_type = blocks[1]
            temp_val = blocks[2].split(";")
            if (data_type == "float"):
                val = [float(v) for v in temp_val]
            elif (data_type == "int"):
                val = [int(v) for v in temp_val]
            else:
                val = temp_val
        dict_[key] = val
    print("dict_ from settings file is ", dict_)
    return dict_  

def parse_compile_file_type2(path):
    """
    Read type II of compilation settings (header file gives
    names of settings, each row is a configuration).
    """
    df = pd.read_csv(path)
    configs = df.columns
    vary_options = []
    for index, row in df.iterrows():
        dict_ = {}
        for config in configs:
            dict_[config] = row[config]
        vary_options.append(dict_)
    return vary_options
    
## ------------------------------------------------------------------
# make save directories and experiment names

def strToList(sampleNonPartitionStr):
    """
    Inputs:
        sampleNonPartitionStr: str, format ("means,sds,concentration")
    """
    sampleNonPartitionList = sampleNonPartitionStr.split(",")
    return sampleNonPartitionList

# def suffixToDict(dataSuffix):
#     """
#     Convert dataSuffix to relevant data identifiers.
    
#     dataSuffix: str, "Ndata=x_D=y_scale=z" for instance.
#         x, y, z
#     """
    
#     dict_ = {}
#     keyValues = dataSuffix.split("_")
#     for pair in keyValues:
#         key, value = pair.split("=")
#         if (key == "Ndata") or (key == "N") or (key == "D") or (key == "degree"):
#             dict_[key] = int(value)
#         else:
#             dict_[key] = float(value)
#     return dict_

# change this file to reflect how options now consists of different args dict()
# that might have different keys


def makeModelArgsDir(modelType, modelArgs):
    dirName = ""
    if (modelType == "mixture"):
        dirName += "sampleNonPartitionStr=%s" %(modelArgs["sampleNonPartitionStr"]) 
        dirName += "_sd1OverSd0=%.2f" % (modelArgs["sd1OverSd0"]) 
        dirName += "_separateSd=%s" %(modelArgs["separateSd"]) + "/"

        # initial DPMM settings and partition initialization
        DPsettings = "sd0=%.2f_sd1=%.2f_alpha=%.2f" %(modelArgs["sd0"], modelArgs["sd1"], modelArgs["alpha"])
        dirName += DPsettings
    else:
        dirName += "nColors=%d" %modelArgs["nColors"] 

    dirName += "/"
    return dirName

def makeCouplingStr(couplingArgs):
    cStr = ""
    cStr += "GibbsCoupling=%s" %(couplingArgs["GibbsCoupling"])
    if ("metric" in couplingArgs):
        cStr += "_metric=%s" %couplingArgs["metric"]
    cStr += "_burnIn=%d_minIter=%d" %(couplingArgs["burnIn"], couplingArgs["minIter"])
    return cStr

def makeSamplerArgsDir(samplerName, samplerArgs):
    """
    Inputs:
        samplerName: str,
        samplerArgs: dict, 

    Output:
        dirName
    """

    # sampler name plus initialization 
    
    dirName = samplerName 
    
    if "fastAssignment" in samplerArgs:
        dirName += "_fastAssignment=%s" %(samplerArgs["fastAssignment"])
    
    dirName += "_initType=%s" %(samplerArgs["initType"]) + "/"

    if "splitMergeDict" in samplerArgs:
        splitMergeDict = samplerArgs["splitMergeDict"]
        dirName += "numRes=%d_numProp=%d_numScan_%d" %(splitMergeDict["numRes"], splitMergeDict["numProp"], 
                                                        splitMergeDict["numScan"])
        dirName +=  "/"

    # coupling type if relevant and time budget
    dirName += "maxTime=%.2f_" %samplerArgs["maxTime"]
    if (samplerArgs["estType"] == "coupled"):
        dirName += makeCouplingStr(samplerArgs["couplingArgs"])
    else:
        singleArgs = samplerArgs["singleArgs"]
        dirName += "loadTime=%s_maxIter=%d" %(singleArgs["loadTime"], singleArgs["maxIter"])
        if (singleArgs["loadTime"]):
            dirName += "_timeReference="
            dirName += makeCouplingStr(samplerArgs["couplingArgs"])
    dirName += "/"

    return dirName 

def makeDirAndFileName(options):
    """
    dirName contains information about data set, type of sampler (Gibbs or Jain Neal),
    whether we do tripleMode, what kind of coupling do we use, what function do we compute the
    expectation of.
    """
    # kind of data set
    dataname = options.dataName
    # if (len(options.dataSuffix) > 0):
        # dataname += "_" + options.dataSuffix

    dirName = options.resultsDir + dataname + "/"

    dirName += makeModelArgsDir(options.modelType, options.modelArgs)

    # what function 
    dirName += "hType=%s/" %options.funcArgs["hType"]

    dirName += makeSamplerArgsDir(options.samplerName, options.samplerArgs)

    # number of replicas
    dirName += "nRep=%d/" %(options.nRep)

    # file names
    estFileName = dirName + "est_taskID=%d_nTasks=%d.csv" %(options.taskID, options.nTasks)
    auxFileName = dirName + "trace_taskID=%d_nTasks=%d.csv" %(options.taskID, options.nTasks)
   
    return dirName, estFileName, auxFileName

def makeHeaderLine(M):
    """
    Input:
        M: scalar, number of dimensions in a hFunction

    header line has format "seed,est0,est1,..,estM,hasMet,tau,nStep,nXstep,timeTaken"
    """
    estsPart = ["est%d" %x for x in range(M)]
    estsStr = ','.join(estsPart)
    line = "seed," +  estsStr + ",hasMet,tau,nStep,nXstep,timeTaken"
    return line

def resultsToLine(seed, 
                  est, 
                  hasMet, tau, nStep, nXstep, 
                  timeTaken):
    """
    Convert estimation results to string, taking 8 digits after decimal.

    Input: 
        est: (M,) list or array, estimates for hFunction
        tau: scalar, meeting time
        nStep: scalar, total number of single transitions
        nXstep: scalar, number of transitions made by X chain
        timeTaken: scalar, time to generate estimate
    """
    estsPart = ["%.8f" %x for x in est]
    estsStr = ",".join(estsPart)
    line = "%d," %(seed) + estsStr + ",%s,%d,%d,%d,%.2f" % (hasMet, tau, nStep, nXstep, timeTaken)
    return line
