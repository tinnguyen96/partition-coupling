"""
Load gene expression data
"""
import pandas as pd
import numpy as np
import pickle

def make_experiment_name(dataroot, Ndata, D, sd,sd0, alpha):
    savedir = make_gene_name(dataroot, Ndata, D)[1] + "/sd=%.2f_sd0=%.2f_alpha=%.2f" %(sd,sd0, alpha)
    return savedir

def make_gene_name(dataroot, Ndata, D):
    name = "%s_data_Ndata=%d_D=%d" %(dataroot, Ndata, D)
    name_as_csv = name + ".csv"
    return (name_as_csv, name)

def load_gene_data(dataroot, directory, Ndata, D):
    savename = directory + make_gene_name(dataroot, Ndata, D)[0]
    print("Will load data from %s" %savename)
    df = pd.read_csv(savename)
    # check if data has a true-label column
    if "z_true_subset" in df.columns:
        print("Unique labels in the data are")
        print(np.unique(df["z_true_subset"]))
        df.drop(df.columns[[0,1]],axis=1,inplace=True)
    else:
        df.drop(df.columns[[0]],axis=1,inplace=True)
    data = df.to_numpy()
    assert data.shape[0] == Ndata
    assert data.shape[1] == D
    return data
