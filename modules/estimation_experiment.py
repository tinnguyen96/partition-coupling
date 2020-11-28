"""estimation_experiment.py runs the processes which compute the parallel
estimates reported in figure 2B.

This runs on single cell RNA-Seq data released by (1) as processed and
modeled by (2).

See runtime argument flags below for details.

References:
(1) Amit Zeisel, Ana B Mun ̃oz-Manchado, Simone Codeluppi, Peter L ̈onnerberg, Gioele La Manno, Anna Jur ́eus, Sueli Marques, Hermany Munguba, Liqun He, Christer Betsholtz, et al. Cell types in the mouse cortex and hippocampus revealed by single-cell RNA-seq. Science, 347(6226):1138–1142, 2015.

(2) Sandhya Prabhakaran, Elham Azizi, Ambrose Carr, and Dana Peer. Dirichlet process mixture model for correcting technical variation in single-cell gene expression data. In International Conference on Machine Learning, pages 1070–1079, 2016.
"""

# Standard libraries
import argparse
from multiprocessing import Pool
import os
import pickle
import numpy as np
from scipy import stats
import sys
import time
try:
    clock = time.clock
except AttributeError:
    clock = lambda : time.clock_gettime(1)

# our implementation
sys.path.append("./modules/")
import utils
from sampling import gibbs_sweep_single
from estimate_utils import prop_in_k_clusters, usual_MCMC_est_crp, unbiased_est_crp , crp_prior
from gene_data import load_gene_data, make_gene_name, make_experiment_name

def parse_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(est_type="estimates", data_dir = "../data/",
            results_dir="../estimation_results/", data_root="gene",
            Ndata=200, D=50, max_iter=1000, max_time=300, sd=5.0, sd0=5.0,
            alpha=0.5, k=10, m=20, init_type="all_same", save_truth_states=False)
    ## simulation
    parser.add_argument("--est_type", type=str, dest="est_type",
                    help="type of estimator", choices=["truth", "coupled", "single"])
    parser.add_argument("--h_type", type=str, dest="h_type",
                    help="type of function")
    parser.add_argument("--data_dir", type=str, dest="data_dir",
                    help="root directory containing data files")
    parser.add_argument("--data_root", type=str, dest="data_root",
                    help="type of data (synthetic or gene expression data)")
    parser.add_argument("--pool_size", type=int, dest="pool_size",
            required=True, help="number of processes to run in parallel")
    parser.add_argument("--n_replicates", type=int, dest="n_replicates",
            default=1, help="number of times to repeat the experiment")
    parser.add_argument("--Ndata", type=int, dest="Ndata",
                    help="number of observations")
    parser.add_argument("--D", type=int, dest="D",
                    help="number of features")
    parser.add_argument("--max_iter", type=int, dest="max_iter",
                    help="maximum number of sweeps through data when computing truth")
    parser.add_argument("--max_time", type=int, dest="max_time",
                    help="maximum processor time to run each replicate")
    parser.add_argument("--sd", type=float, dest="sd",
                    help="std of observational likelihood")
    parser.add_argument("--sd0", type=float, dest="sd0",
                    help="std of prior distribution over cluster means")
    parser.add_argument("--alpha", type=float, dest="alpha",
                    help="concentration of Dirichlet parameter to generate cluster weights")
    parser.add_argument("--k", type=int, dest="k",
                    help="length of burn-in period")
    parser.add_argument("--m", type=int, dest="m",
                    help="minimum number of sweeps to perform when coupling")
    parser.add_argument("--init_type", type=str, dest="init_type",
                    help="how to initialize the Gibbs sampler",
                    choices=['crp_prior', 'all_same'])
    parser.add_argument("--save_truth_states", action="store_true", dest="save_truth_states",
                    help="whether to save states of the long Markov chain")

    options = parser.parse_args()
    return options

options = parse_args()

Ndata, D = options.Ndata, options.D
sd, sd0, alpha = options.sd, options.sd0, options.alpha
init_type = options.init_type
print("init_type: ", init_type)
save_truth_states = options.save_truth_states
results_dir = options.results_dir
k = options.k # burn-in
m = options.m # minimum iterations
pool_size = options.pool_size

if (options.data_root == "gene"):
    data = load_gene_data(options.data_dir, options.Ndata, options.D)
else:
    print("Currently not supported")
    assert False

print(options)

savedir = results_dir + make_experiment_name(Ndata, D, sd,sd0, alpha)

if not os.path.exists(savedir):
    print("Will make directory %s" %savedir)
    os.makedirs(savedir)

def run_rep(est_type, h, maxIter, time_budget, pi0_its=None):
    """
    Only keep track of time to traverse the posterior, not time
    to actually evaluate the function.

    Input:
        est_type: "truth", "single", "coupled"
        h: lambda function
        maxIter: scalar
        time_budget: scalar

    Return
        states: list, states of the Gibbs sampler after sweeps over whole dataset
        result: scalar, estimate of the integral of interest
        num_sweeps: scalar, number of Gibbs sweeps made in the allocated time
            (only interesting for est_type = "truth" or "single")
        num_est: number of estimators generated in the time_budget
            (only interesting for est_type = "coupled")
        s0_state: state of random seed (for later replicability)
    """
    np.random.seed() # seed = None actually means we use the most randomness across processors
    s0_state = np.random.get_state()

    # don't care about time_budget, just sample for long to get estimate
    # of truth
    if est_type == "truth":
        states, result = usual_MCMC_est_crp(k, h, maxIter, data, sd, sd0, alpha, init_type)
        if not (save_truth_states):
            states = None # save memory
        num_sweeps = maxIter
        num_est = 1

    elif est_type == "single":
        if init_type == "all_same":
            X = [np.zeros(data.shape[0], dtype=int)]
        else:
            assert init_type == "crp_prior"
            X = [crp_prior(data.shape[0],alpha)]
        st = clock() # hopefully this is process-specific time
        num_sweeps = 0
        while True:
            X.append(gibbs_sweep_single(data, X[-1].copy(), sd, sd0, alpha))
            num_sweeps += 1
            time_elasped = clock() - st
            if (time_elasped >= time_budget):
                break
        # with reasonable time_budget, this shouldn't happen, but just in case
        if (len(X) <= k):
            states = None
            result = None
            num_est = 0
        else:
            states = X
            ests = [h(x) for x in X[k:]]
            result = np.mean(ests, axis=0)
            num_est = 1

    # as long as time budget hasn't ended, keep generating unbiased estimators
    else:
        assert est_type == "coupled"
        states = []
        ests_ub = []
        num_sweeps = None
        num_est = 0
        st = clock() # hopefully this is process-specific time
        time_left = time_budget
        while True:
            est, X, Y, tau, _ = unbiased_est_crp(k, h, m, data, sd, sd0,
                    alpha, time_left, pi0_its, init_type, coupling)
            if tau is None: break
            states.append((X,Y,tau))
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

    return states, result, num_sweeps, num_est, s0_state

pi0_its = 1
coupling='Optimal'

topK = 10
h_topK = lambda z: prop_in_k_clusters(z, k=topK)

def simulate_and_save(i):
    result = run_rep(options.est_type, h_topK, options.max_iter,
            options.max_time, pi0_its=pi0_its)
    if options.est_type == "coupled":
        savepath = savedir + "/%s_hType=topK_initType=%s_maxTime=%d_burnin=%d_minIter=%d_rep=%04d.pkl"%(
                options.est_type, init_type, options.max_time,k,m, i)
        savepath_est = savedir + "/%s_hType=topK_est_initType=%s_maxTime=%d_burnin=%d_minIter=%d_rep=%04d.pkl"%(
                options.est_type, init_type, options.max_time,k,m, i)
    else:
        assert options.est_type == "single"
        savepath = savedir + "/%s_hType=topK_initType=%s_maxTime=%d_burnin=%d_rep=%04d.pkl"%(
                options.est_type, init_type, options.max_time, k, i)
        savepath_est = savedir + "/%s_hType=topK_est_initType=%s_maxTime=%d_burnin=%d_rep=%04d.pkl"%(
                options.est_type, init_type, options.max_time,k, i)
    with open(savepath, 'wb') as f: pickle.dump(result, f)
    with open(savepath_est, 'wb') as f: pickle.dump(result[1], f)

def simulate_truth(_):
    result = run_rep("truth", h_topK, options.max_iter, options.max_time,
            pi0_its=None)
    return result

st = time.time()

if options.est_type == "truth":
    with Pool(pool_size) as p:
        results = p.map(simulate_truth, range(pool_size))
    savepath = savedir + "/truth_hType=topK_initType=%s_maxIter=%d_savedStates=%s.pkl" %(init_type, options.max_iter, save_truth_states)
    print("Will save results to %s\n" %savepath)
    with open(savepath, 'wb') as f:
        pickle.dump(results, f)
else:
    for rep in range(options.n_replicates):
        with Pool(pool_size) as p:
            p.map(simulate_and_save, range(rep*pool_size, (rep+1)*pool_size))
print("Wall-clock time elasped in minutes %.2f" %((time.time()-st)/60))
