"""
Parse in command-line arguments and run a number of coupling experiments.
(Each root-sim-seed yields num_seeds_from_root meeting time experiments).

"""

## standard libraries
import argparse
import pickle
import os
import numpy as np

## our code
import sampling
from gmm_data import make_gmm_name, load_gmm_data

num_seeds_from_root = 100

def make_experiment_name(sim_seed, options):
    """
    
    """
    root = options.results_dir
    root_with_data = root + make_gmm_name(options.Ndata, options.alpha, options.sd, options.sd0, options.K, seed=options.data_seed)[1] + "/"
    if not os.path.exists(root_with_data):
        print("Will make directory %s" %root_with_data)
        try:
            os.makedirs(root_with_data)
        except:
            print("Another experiment probably created the dir already.")
    train_name = "simseed=%d_maxIter=%d_coupType=%s_lag=%d" %(sim_seed, options.max_iter, options.coupling_type, options.lag)
    full_name = root_with_data + train_name
    full_name_as_dat = full_name + ".dat"
    return (full_name_as_dat, full_name)

def make_compilation_name(options):
    """
    Almost identical to make_experiment_name except we don't store the sim_seed since 
    compiling results for different seeds.
    """
    root = options.results_dir
    root_with_data = root + make_gmm_name(options.Ndata, options.alpha, options.sd, options.sd0, options.K, seed=options.data_seed)[1] + "/"
    if not os.path.exists(root_with_data):
        print("Will make directory %s" %root_with_data)
        try:
            os.makedirs(root_with_data)
        except:
            print("Another experiment probably created the dir already.")
    train_name = "maxIter=%d_coupType=%s_lag=%d" %(options.max_iter, options.coupling_type, options.lag)
    full_name = root_with_data + train_name
    full_name_as_png = full_name + ".png"
    return full_name_as_png

def parse_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(root_sim_seed=0, max_iter=100, coupling_type="Optimal", lag=20, data_seed=0,Ndata=20, sd=2.0, sd0=2.0, alpha=0.5, K=3, data_dir='data/', results_dir="results/")
    ## simulation
    parser.add_argument("--root_sim_seed", type=int, dest="root_sim_seed",
                    help="to generate random seed for randomness in Gibbs moves")
    parser.add_argument("--max_iter", type=int, dest="random_seed",
                    help="maximum number of sweeps through all observations")
    parser.add_argument("--coupling_type", type=str, dest="coupling_type",
                    help="kind of coupling of Gibbs moves")
    parser.add_argument("--lag", type=int, dest="lag",
                    help="how many steps to move one chain forward before starting the coupling")
    ## data
    parser.add_argument("--data_seed", type=int, dest="data_seed",
                    help="random seed used to generate dataset")
    parser.add_argument("--Ndata", type=int, dest="Ndata",
                    help="cardinality of dataset")
    parser.add_argument("--sd", type=float, dest="sd",
                    help="std of observational likelihood")
    parser.add_argument("--sd0", type=float, dest="sd0",
                    help="std of prior distribution over cluster means")
    parser.add_argument("--alpha", type=float, dest="alpha",
                    help="concentration of Dirichlet parameter to generate cluster weights")
    parser.add_argument("--K", type=int, dest="K",
                    help="number of clusters in GMM")
    ## organizational
    parser.add_argument("--data_dir", type=str, dest="data_dir",
                    help="folder containing the data files")
    parser.add_argument("--results_dir", type=str, dest="results_dir",
                    help="folder containing the data files")
    
    options = parser.parse_args()
    return options 

def simulate(options):
    sim_seeds = range(options.root_sim_seed*num_seeds_from_root, (options.root_sim_seed+1)*num_seeds_from_root)
    # print(sim_seeds)
    for sim_seed in sim_seeds:
        print("Currently working on seed %d" %sim_seed)

        ## the random seed step is very important! 
        np.random.seed(sim_seed)

        ## if coupling successful, how much time?
        savepath = make_experiment_name(sim_seed, options)[0]
        print("Will save result at %s" %savepath)
        log_file = open(savepath, "w")
        log_file.write("sim_seed lag\n")
        log_file.write("%d %d\n\n" %(sim_seed,options.lag))

        ## load data
        data = load_gmm_data(options.data_dir, options.Ndata, options.alpha, options.sd, options.sd0, options.K, options.data_seed)

        ## forward one chain, keeping the other one fixed
        initz1, initz2 = sampling.forward_one_chain(options.lag, data, options.sd, options.sd0, options.alpha)

        ## couple the two chains
        final_state, has_met, total_iters = sampling.crp_gibbs_couple(data, options.sd, options.sd0, initz1.copy(), initz2.copy(), options.alpha, log_freq=None, maxIters=options.max_iter, coupling=options.coupling_type, save_base=None)

        log_file.write("has_met iters_taken max_iter\n")
        log_file.write("%s %d %d" %(has_met, total_iters, options.max_iter))
        log_file.flush()
        log_file.close()
        
        print("Finished with one seed ----------------------------------------------------")
    print("Finished with all seeds from root ----------------------------------------------------")
    return 

if __name__ == "__main__":
    options = parse_args()
    simulate(options)