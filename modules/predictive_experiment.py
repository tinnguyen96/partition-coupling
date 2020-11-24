"""
Report the predictive density for DP posterior over gmm data.
"""

# Standard libaries 
import argparse
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_context('paper', rc={'xtick.labelsize': 15, 'ytick.labelsize': 15, 'lines.markersize': 5})
sb.set_style('whitegrid')
import numpy as np
np.set_printoptions(precision=2)
from scipy import stats
import imp
import time
try:
    clock = time.clock
except AttributeError:
    clock = lambda : time.clock_gettime(1)
import pickle

# our implementation 
import utils
from sampling import gibbs_sweep_single
from estimate_utils import prop_in_k_clusters, usual_MCMC_est_crp, unbiased_est_crp, crp_prior, posterior_predictive_density_1Dgrid
import gmm_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(data_dir = "../data/", results_dir="../estimation_results_postBug/", is_nested=False, Ndata=500, data_sd=2.0, data_sd0=10.0, data_K=10, data_alpha=0.5, data_seed=0, max_time=100, sd=2.0, sd0=3.0, alpha=1, init_type="crp_prior", pool_size=100, num_replicates=100)
    ## info about data
    parser.add_argument("--data_dir", type=str, dest="data_dir",
                    help="root directory containing data files")
    parser.add_argument("--Ndata", type=int, dest="Ndata",
                    help="number of observations")
    parser.add_argument("--data_sd", type=float, dest="data_sd",
                    help="std of observational likelihood that generated gmm data")
    parser.add_argument("--data_sd0", type=float, dest="data_sd0",
                    help="std of prior distribution over cluster means that generated gmm data")
    parser.add_argument("--data_alpha", type=float, dest="data_alpha",
                    help="concentration of Dirichlet parameter to generate cluster weights that generated gmm data")
    parser.add_argument("--data_seed", type=float, dest="data_seed",
                    help="random seed that generated gmm data")
    ## info about estimator
    parser.add_argument("--max_time", type=int, dest="max_time",
                    help="maximum processor time to run each replicate")
    parser.add_argument("--max_iter", type=int, dest="max_iter",
                    help="maximum number of sweeps through data when computing truth")
    parser.add_argument("--sd", type=float, dest="sd",
                    help="std of observational likelihood")
    parser.add_argument("--sd0", type=float, dest="sd0",
                    help="std of prior distribution over cluster means")
    parser.add_argument("--alpha", type=float, dest="alpha",
                    help="concentration of Dirichlet parameter to generate cluster weights")
    parser.add_argument("--init_type", type=str, dest="init_type",
                    help="how to initialize the Gibbs sampler")
    parser.add_argument("--is_nested", action="store_true", dest="is_nested",
                    help="whether the gmm data was generated with the nested version")
    ## info about multi-processing
    parser.add_argument("--pool_size", type=int, dest="pool_size",
                    help="how many jobs in parallel to run for each replicate")
    parser.add_argument("--num_replicates", type=int, dest="num_replicates",
                    help="how many replicates")
    options = parser.parse_args()
    return options 

options = parse_args()
print(options)

Ndata = options.Ndata
D = 1
sd, sd0, alpha = options.sd, options.sd0, options.alpha
init_type = options.init_type
is_nested = options.is_nested

data, grid_and_density = gmm_data.load_gmm_data(options.data_dir, D, Ndata, options.data_alpha, options.data_sd, options.data_sd0, options.data_K, options.data_seed, is_nested)
savedir = options.results_dir + gmm_data.make_experiment_name(Ndata, D, options.data_sd, options.data_sd0, options.data_alpha, options.data_K, options.data_seed, is_nested, sd, sd0, alpha)

if not os.path.exists(savedir):
    print("Will make directory %s" %savedir)
    os.makedirs(savedir)
    
## estimate predictive density using coupled chains
def run_rep(h, maxIter, time_budget):
    """
    Input:
        h: lambda function
        maxIter: scalar 
        time_budget: scalar
        
    Return 
    """
    np.random.seed() # seed = None actually means we use the most randomness across processors
    s0_state = np.random.get_state()
    ests_ub = []
    num_sweeps = None
    num_est = 0
    st = clock() # hopefully this is process-specific time
    time_left = time_budget
    while True:
        est, X, Y, tau, _ = unbiased_est_crp(k, h, m, data, sd, sd0,
                alpha, time_left, pi0_its, init_type, coupling)
        if tau is None: break
        ests_ub += [est]
        time_left = time_budget - (clock() - st)
        num_est += 1
        if time_left <= 0: break
    # this is unlikely to happen if we set time_budget to be reasonable, but
    # just in case
    if (num_est == 0):
        result = None
    else:
        # average the result of successful meetings
        result = np.mean(ests_ub, axis=0)

    return 0, result, num_sweeps, num_est, s0_state

k = 10 # burn-in
m = 100 # minimum iterations
pi0_its = 5
coupling='Optimal'

pool_size = options.pool_size
num_reps = options.num_replicates
grid = grid_and_density[0] # evaluate posterior predictive where the synthetic data function evaluates

h_predictive = lambda z: posterior_predictive_density_1Dgrid(grid, data, z, sd, sd0, alpha)

def simulate(_):
    result = run_rep(h_predictive, options.max_iter, options.max_time)
    print("completed pool job")
    return result
    
st = time.time()

results = []
for i in range(num_reps):
    with Pool(pool_size) as p:
        rep_results = p.map(simulate, range(pool_size))
    results.extend(rep_results)
    print("completed replicate number %d" %i)

total_count = num_reps*pool_size 
    
print("Wall-clock time elasped in minutes %.2f" %((time.time()-st)/60))

savepath = savedir + "/coupled_estimates_totRep=%d_hType=predictive_initType=%s_maxTime=%d_burnin=%d_minIter=%d.pkl" %(total_count, init_type, options.max_time,k,m)
print("Will save results to %s" %savepath)
print()
with open(savepath, 'wb') as f:
    pickle.dump(results, f)