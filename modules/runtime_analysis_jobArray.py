"""
Compare the processor time versus number of sweeps for single-chain
versus coupled-chains.
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
from time import time
import time
import pickle

# our implementation 
import utils
from sampling import gibbs_sweep_single
from estimate_utils import prop_in_k_clusters, usual_MCMC_est_crp, unbiased_est_crp , crp_prior
from gene_data import load_gene_data, make_gene_name, make_experiment_name

# hyper-parameters combiniations. Should use some product function 
# combos = [[2.0, 3.0, 0.5], [2.0, 5.0, 0.5], [2.0, 7.0, 0.5], [2.0, 9.0, 0.5], [3.0,2.0,1],[5.0,5.0,1],[7.0, 5.0,1],[9.0, 2.0,1]]

combos = [[3.0,3.0,1],[3.0,4.0,1],[3.0, 5.0,1],[3.0, 6.0, 1],
         [3.0,3.0,2],[3.0,4.0,2],[3.0, 5.0,2],[3.0, 6.0, 2]]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(data_dir = "../data/", combo_index=0, results_dir="../estimation_results_postBug/", data_root="gene_withLabels_randomSeed=0", Ndata=200, D=50, max_time=300, init_type="crp_prior", save_truth_states=False)
    ## simulation
    parser.add_argument("--data_dir", type=str, dest="data_dir",
                    help="root directory containing data files")
    parser.add_argument("--combo_index", type=int, dest="combo_index",
                    help="index into combos list of hyper-params settings")
    parser.add_argument("--results_dir", type=str, dest="results_dir",
                    help="where to store results")
    parser.add_argument("--data_root", type=str, dest="data_root",
                    help="type of data (synthetic or gene expression data)")
    parser.add_argument("--Ndata", type=int, dest="Ndata",
                    help="number of observations")
    parser.add_argument("--D", type=int, dest="D",
                    help="number of features")
    parser.add_argument("--max_time", type=int, dest="max_time",
                    help="maximum processor time to run each replicate")
    parser.add_argument("--max_iter", type=int, dest="max_iter",
                    help="maximum number of sweeps through data when computing truth")
    parser.add_argument("--init_type", type=str, dest="init_type",
                    help="how to initialize the Gibbs sampler")
    parser.add_argument("--save_truth_states", action="store_true", dest="save_truth_states",
                    help="whether to save states of the long Markov chain")
    
    options = parser.parse_args()
    return options 

options = parse_args()

print(options)
Ndata, D = options.Ndata, options.D
combo = combos[options.combo_index]
sd, sd0, alpha = combo[0], combo[1], combo[2]
init_type = options.init_type
save_truth_states = options.save_truth_states

data = load_gene_data(options.data_root, options.data_dir, options.Ndata, options.D)
savedir = options.results_dir + make_experiment_name(options.data_root, Ndata, D, sd,sd0, alpha)

if not os.path.exists(savedir):
    print("Will make directory %s" %savedir)
    os.makedirs(savedir)
    
def run_rep(est_type, h, maxIter, time_budget):
    """
    Input:
        est_type: "truth", "single", "coupled"
        h: lambda function
        maxIter: scalar 
        time_budget: scalar
        
    Return 
        list_elapsed_time_list: list of tuples, 
            first element of tuple is list of processor time elapsed after sweeps
            second element, whether meeting time has happened
        s0_state: state of random seed used for the replicate (for later replicability)
    """
    np.random.seed() # seed = None actually means we use the most randomness across processors
    s0_state = np.random.get_state()
    
    if (est_type == "single"):
        elapsed_time_list = []
        if (init_type == "all_same"):
            X = [np.zeros(data.shape[0], dtype=int)]
        elif (init_type == "crp_prior"):
            X = [crp_prior(data.shape[0],alpha)]
        st = time.clock() # hopefully this is process-specific time
        num_sweeps = 0
        while (True):
            X.append(gibbs_sweep_single(data, X[-1].copy(), sd, sd0, alpha))
            num_sweeps += 1
            time_elasped = time.clock() - st 
            elapsed_time_list.append(time_elasped)
            if (time_elasped >= time_budget):
                break
        list_elapsed_time_list = [(elapsed_time_list,None)]
            
    # as long as time budget hasn't ended, keep generating unbiased estimators
    elif (est_type == "coupled"):
        list_elapsed_time_list = []
        num_sweeps = None
        st = time.clock() # hopefully this is process-specific time
        time_left = time_budget
        while (True):
            est, X, Y, tau, elapsed_time_list = unbiased_est_crp(k, h, m, data, sd, sd0, alpha, time_left, pi0_its, init_type, coupling)
            list_elapsed_time_list.append((elapsed_time_list,tau))
            time_left = time_budget - (time.clock() - st)
            if (time_left <= 0):
                break
                
    return list_elapsed_time_list, s0_state

k = 10 # burn-in
m = 20 # minimum iterations
pi0_its = 5
coupling='Optimal'
quarter_size = 10
half_size = 20

topK = 10
h_topK = lambda z: prop_in_k_clusters(z, k=topK)

def simulate_short(_):
    result = run_rep("coupled", h_topK, options.max_iter, options.max_time)
    print("complete")
    return result
    
st = time.time()

with Pool(quarter_size) as p:
    results = p.map(simulate_short, range(quarter_size))

print("Wall-clock time elasped in minutes %.2f" %((time.time()-st)/60))

savepath = savedir + "/runtimes_estimates_hType=topK_initType=%s_maxTime=%d_burnin=%d_minIter=%d.pkl" %(init_type, options.max_time,k,m)
print("Will save results to %s" %savepath)
print()
with open(savepath, 'wb') as f:
    pickle.dump(results, f)
        