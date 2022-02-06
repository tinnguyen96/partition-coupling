# join small csv files into one big csv and delete small files

import os
import json
import names_and_parser
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools

USE_SHORT_DIR=True # good for compiling results
REMOVE_CSV=False # good for memory

def compressTraceFiles(options):
    dirName, _, _ = names_and_parser.makeDirAndFileName(options)
    nTasks = options.nTasks
    
    auxFiles = []
    
    for taskID in tqdm(range(nTasks)):
        auxFileName = dirName + "trace_taskID=%d_nTasks=%d.csv" %(taskID, nTasks)
        with open(auxFileName, "rb") as file:
            auxFiles.append(json.load(file))
        if (False):
            os.remove(auxFileName)
    
    with open(dirName + "traces.json", "w") as outFile:
        json.dump(auxFiles, outFile, indent=4)
        
    if USE_SHORT_DIR:
        if not os.path.exists(options.shortDir):
            print("Will make directory %s" %options.shortDir)
            os.makedirs(options.shortDir)
        with open(options.shortDir + "traces.json", "w") as outFile:
            json.dump(auxFiles, outFile, indent=4)
    return 

def compressGroundTruthCsv(options):
    dirName, _, _ = names_and_parser.makeDirAndFileName(options)
    nTasks = options.nTasks
    
    truthDfs = []
    
    for taskID in tqdm(range(nTasks)):
        truthName = dirName + "est_taskID=%d_nTasks=%d.csv" %(taskID, nTasks)
        truthDfs.append(pd.read_csv(truthName))
        if (REMOVE_CSV):
            os.remove(truthName)
    
    combinedTruth = pd.concat(truthDfs)
    print(combinedTruth)
    estColumns = combinedTruth.filter(regex=("est*"))
    
    groundTruth = estColumns.mean(axis=0)
    sem = estColumns.sem(axis=0)
    semAsPercentageError = 100*(sem/groundTruth)
    
    if (options.funcArgs["hType"] == "CC"):
        nObs = int(np.sqrt(len(estColumns.columns)))
        CCIndices = options.funcArgs["CCIndices"]
        
        twoDIndices = list(itertools.product(CCIndices, CCIndices))
        tupleTwoDIndices = list(zip(*twoDIndices))
        oneDIndices = np.ravel_multi_index(tupleTwoDIndices, [nObs, nObs])
        
        subsetGroundTruth = groundTruth[oneDIndices]
        subsetSem = sem[oneDIndices]
        subsetSEMAsPercentageError = semAsPercentageError[oneDIndices]
        print("subsetGroundTruth", subsetGroundTruth)
        print("subsetSem", subsetSem)
        print("subsetSEMAsPercentageError", subsetSEMAsPercentageError)
    else:
        print(groundTruth)
        print(sem)
        print(semAsPercentageError)
        
    overallDf = pd.concat([groundTruth, sem, semAsPercentageError], axis=1)
    overallDf.columns = ["groundTruth", "sem", "semAsPercentageError"]
    
    overallDf.to_csv(dirName + "groundTruth.csv", index=False)
    if USE_SHORT_DIR:
        if not os.path.exists(options.shortDir):
            print("Will make directory %s" %options.shortDir)
            os.makedirs(options.shortDir)
        overallDf.to_csv(options.shortDir + "groundTruth.csv", index=False)
    
    return 
    
def compressEstimatesCsv(options):
    dirName, _, _ = names_and_parser.makeDirAndFileName(options)
    nTasks = options.nTasks
    estFiles = []
    
    for taskID in tqdm(range(nTasks)):
        estFileName = dirName + "est_taskID=%d_nTasks=%d.csv" %(taskID, nTasks)
        estFiles.append(pd.read_csv(estFileName))
        if (REMOVE_CSV):
            os.remove(estFileName)
        
    combinedEst = pd.concat(estFiles)
    
    combinedEst.to_csv(dirName + "estType=%s.csv" %options.samplerArgs["estType"], index=False)
    if USE_SHORT_DIR:
        if not os.path.exists(options.shortDir):
            print("Will make directory %s" %options.shortDir)
            os.makedirs(options.shortDir)
        combinedEst.to_csv(options.shortDir + "estType=%s.csv" %options.samplerArgs["estType"], index=False)
        
    return 

def compressCsv(options):
    if (options.samplerArgs["estType"]=="truth"):
        compressGroundTruthCsv(options)
    else:
        compressEstimatesCsv(options)

if __name__ == "__main__":
    options = names_and_parser.parse_args()
    
    optionsAsDict = vars(options) # alternate view of options
    with open(options.configPath) as f:
        config = json.load(f)
    optionsAsDict.update(config)
    
    print('===========================')
    for key, val in vars(options).items():
        print('{}: {}'.format(key, val))
    print('===========================')
    
    if (False):
        compressTraceFiles(options)
    
    if (True):
        compressCsv(options)