# Todos
# Incorporate the processing functions from the relevant notebooks

import pandas as pd
import numpy as np

import igraph
import pickle

def loadMixtureData(rDir, dataName):
    """
    Inputs:
        rDir: str
        dataName: str

    Output:
        X: (N,D)
    """
    name = rDir + dataName
    df = pd.read_csv(name)
    X = df.to_numpy()
    print("nObs, nCov =", X.shape)
    return X

def loadGraphData(rDir, dataName):
    path = rDir + dataName
    graph = igraph.Graph.Read_Pickle(open(path, "rb"))
    return graph

def processSeedData(rDir, dataName):
    return 

def makeSyntheticMixtureData(rDir):
    return

def makeGraphData(rDir):
    return 

if __name__ == "__main__":
    sys.exit(0)