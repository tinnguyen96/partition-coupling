"""
Generate and save data from GMM for varying sizes.
"""
## standard libraries
import numpy as np
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_context('paper', rc={'xtick.labelsize': 15, 'ytick.labelsize': 15, 'lines.markersize': 5})
sb.set_style('whitegrid')

def make_experiment_name(Ndata, D, data_sd, data_sd0, data_alpha, data_K, data_seed, is_nested, sd, sd0, alpha):
    savedir = make_gmm_name(D, Ndata, data_alpha, data_sd, data_sd0, data_K, data_seed, is_nested)[1] + "/sd=%.2f_sd0=%.2f_alpha=%.2f" %(sd, sd0, alpha)
    return savedir 

def make_gmm_name(D, Ndata, alpha, sd, sd0, K, seed, is_nested):
    name = "GMM_D=%d_N=%d_alpha=%.2f_sd=%.2f_sd0=%.2f_K=%d_seed=%d" %(D, Ndata, alpha, sd, sd0, K, seed)
    if (is_nested):
        name = "nested" + name
    name_as_pkl = name + ".pkl"
    return (name_as_pkl, name)

def make_gmm_data(directory, D, Ndata, alpha, sd, sd0, K, seed):
    # TRANSLATION OF TAMARA's CODE INTO PYTHON
    #
    # generate Gaussian mixture model data for inference later
    #
    # Args:
    #  Ndata: number of data points to generate
    #  D: number of dimensions of data points
    #  alpha: concentration of Dirichlet parameter to generate cluster weights
    #  sd: covariance matrix of data points around the
    #      cluster-specific mean is [sd^2, 0; 0, sd^2];
    #      i.e. this is the standard deviation in either direction
    #  sd0: std for prior mean
    #  K: number of clusters
    #  seed: random seed; once set, the mixture density is fixed for all Ndata
    #
    # Returns:
    #  x: an Ndata x D matrix of data points
    #  z: an Ndata-long vector of cluster assignments
    #  mu: a K x D matrix of cluster means,
    #      where K is the number of clusters

    # this will seed both np.random and stats distributions
    np.random.seed(seed)
    # matrix of cluster centers: one in each quadrant
    mu = np.random.normal(scale=sd0, size=[K, D])
    # vector of component frequencies
    rho = stats.dirichlet.rvs(alpha=2*np.ones(K))[0]

    # assign each data point to a component
    z = np.random.choice(range(K), p=rho, replace=True, size=Ndata)
    # draw each data point according to the cluster-specific
    # likelihood of its component
    x = mu[z] + np.random.normal(scale=sd, size=[Ndata,D])  # (Ndata, D) array
    
    # make save folders
    savename = directory + make_gmm_name(D, Ndata, alpha, sd, sd0, K, seed)[1]
    savename_as_pkl = savename + ".pkl"
    
    diction = {"data":x}
    
    # visualize
    if (D == 1):
        min_range, max_range = np.min(mu)-sd0, np.max(mu)+sd0, 
        delta = (max_range-min_range)/150
        grid = np.arange(min_range, max_range, delta)
        ppd = np.zeros(grid.shape[0])
        for k in range(K):
            ppd += rho[k]*stats.norm(loc=mu[k], scale=sd).pdf(grid)
        
        plt.figure()
        plt.hist(x, density=True, label='histogram')
        plt.plot(grid, ppd, label="density")
        plt.title("density of GMM, Ndata=%d, K=%d" %(Ndata, K), fontsize=18)
        plt.xlabel("x", fontsize=18)
        plt.legend(fontsize=18)
        savefigpath = savename + ".png"
        print("Will save figure to %s" %savefigpath)
        plt.savefig(savefigpath)
        diction["density"] = (grid, ppd)
        
    print("Will save data to %s" %savename_as_pkl)
    a_file = open(savename_as_pkl, "wb")
    pickle.dump(diction, a_file)
    a_file.close()
    
    return x

def make_nested_gmm_data(Ndata, smallNdata_list, directory, D, alpha, sd, sd0, K, seed):
    """
    Make one gmm data set of length topNdata, and take subsets to 
    mimick collecting more observations from a iid process.
    This function exists because make_gmm_data as it is written
    doesn't preserve the nestedness among different Ndata. It would 
    have been preserved if we used a loop and sample each z and x in 
    a sequential manner.
    """
    # this will seed both np.random and stats distributions
    np.random.seed(seed)
    # matrix of cluster centers: one in each quadrant
    mu = np.random.normal(scale=sd0, size=[K, D])
    # vector of component frequencies
    rho = stats.dirichlet.rvs(alpha=2*np.ones(K))[0]

    # assign each data point to a component
    z = np.random.choice(range(K), p=rho, replace=True, size=Ndata)
    # draw each data point according to the cluster-specific
    # likelihood of its component
    x = mu[z] + np.random.normal(scale=sd, size=[Ndata,D])  # (Ndata, D) array
    
    # evaluate density on grid
    if (D == 1):
        min_range, max_range = np.min(mu)-sd0, np.max(mu)+sd0, 
        delta = (max_range-min_range)/150
        grid = np.arange(min_range, max_range, delta)
        ppd = np.zeros(grid.shape[0])
        for k in range(K):
            ppd += rho[k]*stats.norm(loc=mu[k], scale=sd).pdf(grid)
        
    is_nested = True
    for N in smallNdata_list:
        subset_x = x[range(N)]
        # make save folders
        savename = directory + make_gmm_name(D, N, alpha, sd, sd0, K, seed, is_nested)[1]
        savename_as_pkl = savename + ".pkl"
        diction = {"data":subset_x}
        
        # visualize
        if (D == 1):
            plt.figure()
            plt.hist(subset_x, density=True, label='histogram')
            plt.plot(grid, ppd, label="density")
            plt.title("density of GMM, Ndata=%d, K=%d" %(N, K), fontsize=18)
            plt.xlabel("x", fontsize=18)
            plt.legend(fontsize=18)
            savefigpath = savename + ".png"
            print("Will save figure to %s" %savefigpath)
            plt.savefig(savefigpath)
            diction["density"] = (grid, ppd)
            
        print("Will save data to %s" %savename_as_pkl)
        a_file = open(savename_as_pkl, "wb")
        pickle.dump(diction, a_file)
        a_file.close()

    return 

def load_gmm_data(directory, D, Ndata, alpha, sd, sd0, K, seed, is_nested):
    savename = directory + make_gmm_name(D, Ndata, alpha, sd, sd0, K, seed, is_nested)[0]
    print("Will load data from %s" %savename)
    diction = pickle.load(open(savename, "rb"))
    data = diction["data"]
    print("Loaded data has shape (%d, %d)" %(data.shape[0], data.shape[1]))
    if D == 1:
        density = diction["density"]
    else:
        density = None
    return data, density

if __name__ == "__main__":
    K = 6
    D = 1 
    Ndata = 500
    smallNdata_list = [100, 200, 300, 400, 500]
    sd, sd0, alpha = 2., 10., 0.5
    seed = 0
    directory = "data/"
    #for Ndata in Ndata_list:
    #    make_gmm_data(directory, D, Ndata, alpha, sd, sd0, K, seed)
    make_nested_gmm_data(Ndata, smallNdata_list, directory, D, alpha, sd, sd0, K, seed)
    
