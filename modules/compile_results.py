"""
Compile results for runtime_analysis.py
"""

## standard libraries
from tqdm import tqdm
import argparse
import os
import pickle
import fnmatch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_context('paper', rc={'xtick.labelsize': 15, 'ytick.labelsize': 15, 'lines.markersize': 5})
sb.set_style('whitegrid')

## our code
from meeting_time_experiment import num_seeds_from_root
from estimate_utils import prop_in_k_clusters
from unbiased_estimation import unbiased_est

# ---------------------------------------------------------
# estimation_experiments
def average_rep_truth(truth):
    ests = [truth[i][1] for i in range(0,len(truth))]
    ests = np.array(ests)
    trueval = np.mean(ests, axis=0)
    MSEs = np.mean(np.square(ests-trueval),axis=0)
    return trueval, MSEs

def compute_estimate_error(estimates, est_type, topK, trueval):
    """compute_estimate_error aggregates estimates from multiple processes
    and compare to the "ground truth" from long chains.

    Args:
        estimates: list of estimates from different processes -- each
            estimate is an array of proportions.
    """
    # Take entry only for topK, and remove unsuccessful couplings
    ests = [estimate[topK] for estimate in estimates if estimate is not None]
    print("Mean Estimate is", np.mean(ests))
    print("Final est error [ mean(ests) - true] = ", np.mean(ests) - trueval)
    print("SEM(ests) = ", stats.sem(ests))

def compute_average_single_sweeps(estimates):
    sweeps = [estimate[2] for estimate in estimates]
    avg_sweeps = np.mean(sweeps)
    return avg_sweeps

def compute_average_ppd(results):
    """
    Inputs:
        results: list, typical outcome from predictive_experiment.py
    """
    index_range = range(len(results))
    # extract distribution of meeting times. Get number of replicates that had no meeting
    est_ub = []
    unsuccessful_counts = 0
    for i in index_range:
        replicate = results[i]
        if (replicate[1] is None):
            unsuccessful_counts += 1
        else:
            est_ub.append(replicate[1])
    print("Number of unsuccessful replicates out of total is %d / %d" %(unsuccessful_counts, len(index_range)))
    average_est = np.mean(est_ub, axis=0)
    return average_est, unsuccessful_counts

def compute_num_couplings(estimates):
    num_ests = [estimate[3] for estimate in estimates]
    return num_ests

def help_get_coupling_times(successes):
    ans = [exp[2] for exp in successes]
    return ans

def get_coupling_times(estimates):
    successful_couplings = [estimate[0] for estimate in estimates
            if len(estimate[0]) > 0]
    coupling_times = [help_get_coupling_times(est) for est in successful_couplings]
    coupling_times = np.array(coupling_times)
    return coupling_times

def helper_estimators_topK_vary(successful_couplings, topK, k, m, truthval):
    h = lambda z: prop_in_k_clusters(z, k=10)
    final_ests = []
    for successes in successful_couplings:
        ests = []
        for succ in successes:
            ans = unbiased_est(k, h, m, succ[0], succ[1], succ[2])[topK]
            ests.append(ans)
        ests = np.array(ests)
        final_est = np.mean(ests)
        final_ests.append(final_est)

    est_error, std_error = np.mean(final_ests)-truthval, stats.sem(final_ests)
    return est_error, std_error

def helper_estimators_topK_vary_single_chain(states_by_chain, topK, k, truthval):
    h = lambda z: prop_in_k_clusters(z, k=10)
    final_ests = [np.mean([h(state)[topK] for state in states[k:]]) for states in
            states_by_chain]
    est_error, std_error = np.mean(final_ests)-truthval, stats.sem(final_ests)
    return est_error, std_error

def report_topK_performance(savedirroot, init_type, max_iter, save_saves,
        topK, max_time, k, m, explore_k=True):
    """report_topK_performance looks at performance of single and coupled
    chain estimates relative to long chain MCMC on estimating posterior mean
    proportion of datapoints in the top K clusters.


    Args:
        savedirroot: path to directory containing simulation results.
        init_type: how chains were initialized ('crp_prior' or 'all_same')
        max_iter: integer number of sweeps for long chain MCMC
        save_saves: (BLT - I'm not actually sure what this means/does )
        topK: integer number of top clusters in which to compute membership
            proportions.
        max_time: time allowance in seconds allocated to each process.
        k, m: burn-in and minimum chain length parameters.  m is specific to
            coupled chain estimates.
        explore_k: Boolean, set to True to compare performance for different
            burn-in lengths.

    Returns:
        None
    """
    ## load truth
    truthpath = savedirroot + "/truth_hType=topK_initType=%s_maxIter=%d_savedStates=%s.pkl"%(
            init_type, max_iter, save_saves)
    print("Loading truth from %s" %truthpath)
    truth = pickle.load(open(truthpath, "rb"))
    truths, stderrs = average_rep_truth(truth)
    truthval, stderr = truths[topK], stderrs[topK]
    print("True value of topK proportion for topK = %d is %.5f +- %.5f\n"%(
        topK,truthval, stderr))


    ## load coupled estimates
    estimate_fn_base_coupled = "coupled_hType=topK_est_initType=%s_maxTime=%d_burnin=%d_minIter=%d"%(
        init_type, max_time,k,m)
    print("Loading estimates from %s*" %estimate_fn_base_coupled)
    fns = [ file for file in os.listdir(savedirroot) if fnmatch.fnmatch(file, estimate_fn_base_coupled+"*")]
    estimates_coupled = [pickle.load(open(savedirroot + "/" + fn, "rb")) for fn in fns]
    # compute coupled-chain MSE
    print("Number of procs. no couplings %d / %d\n"%(
        np.sum(estimates_coupled==None),len(estimates_coupled)))
    compute_estimate_error(estimates_coupled, "coupled", topK, truthval)


    ## load single estimates
    estimate_fn_base_single = "single_hType=topK_est_initType=%s_maxTime=%d_burnin=%d"%(
            init_type, max_time,k)
    print("Loading estimates from %s*" %estimate_fn_base_single)
    fns = [ file for file in os.listdir(savedirroot) if fnmatch.fnmatch(file, estimate_fn_base_single+"*")]
    estimates_single = [pickle.load(open(savedirroot + "/" + fn, "rb")) for fn in fns]
    # compute single-chain MSE
    compute_estimate_error(estimates_single, "single", topK, truthval)
    print("--------")

    # Explore difference in single and coupled chain estimates with k
    if not explore_k: return

    ### Load in full chain states and report some summary information
    # For coupled chains
    estimate_fn_base_coupled = "coupled_hType=topK_initType=%s_maxTime=%d_burnin=%d_minIter=%d" %(
        init_type, max_time,k,m)
    print("Loading estimates from %s*" %estimate_fn_base_coupled)
    fns = [ file for file in os.listdir(savedirroot) if fnmatch.fnmatch(file, estimate_fn_base_coupled+"*")]
    estimates_coupled = [pickle.load(open(savedirroot + "/" + fn, "rb")) for fn in fns]
    num_couplings = compute_num_couplings(estimates_coupled)
    print("Number of successful couplings is %d" %np.sum(num_couplings))

    coupling_times = get_coupling_times(estimates_coupled)
    coupling_times_all = []
    for times in coupling_times: coupling_times_all.extend(times)
    print("Meeting times of successful experiments (Mean +- sd)",
            np.mean(coupling_times_all), "+-", np.std(coupling_times_all))
    # get (X,Y,tau) of successful coupling experiments
    successful_couplings = [estimate[0] for estimate in estimates_coupled if len(estimate[0]) > 0]

    # For single chains
    estimate_fn_base_single = "single_hType=topK_initType=%s_maxTime=%d_burnin=%d"%(
            init_type, max_time,k)
    print("Loading estimates from %s*" %estimate_fn_base_single)
    fns = [ file for file in os.listdir(savedirroot) if fnmatch.fnmatch(file, estimate_fn_base_single+"*")]
    estimates_single = [pickle.load(open(savedirroot + "/" + fn, "rb")) for fn in fns]
    avg_sweeps = compute_average_single_sweeps(estimates_single)
    print("Given time budget %.2f, single chain evolved %.2f sweeps (averaged across replicates)" %(max_time,avg_sweeps))

    states_by_chain = [estimate[0] for estimate in estimates_single]

    vary_list = [1, 10, 20, 50, 100] # values of K to compare

    for i in vary_list:
        # fix m, vary k
        est_err, std_err = helper_estimators_topK_vary(successful_couplings, topK, i, m, truthval)
        print("k=%03d\t--- Err +- SE : %f +- %f"%(i, est_err, std_err))

        est_err_single, std_err_single = helper_estimators_topK_vary_single_chain(states_by_chain, topK, i, truthval)
        print("single\t--- Err +- SE : %f +- %f\n"%(est_err_single, std_err_single))
        
def report_predictive_density(Ndata, tot_reps, savedirroot, true_density, init_type, max_time, k, m):
    
    h_type = "predictive"
    # load estimates
    estimatespath = savedirroot + "/coupled_estimates_totRep=%d_hType=%s_initType=%s_maxTime=%d_burnin=%d_minIter=%d.pkl" %(tot_reps, h_type, init_type, max_time,k,m)
    print("Loading estimates from %s" %estimatespath)
    estimates = pickle.load(open(estimatespath, "rb"))
    print("Finished loading estimates")
    average_ppd, unsuccessful_counts = compute_average_ppd(estimates)
    
    # overlay posterior predictive density with true density
    assert np.allclose(true_density[0],average_ppd[0,:])
    L1_diff = np.sum(np.abs(true_density[1]-average_ppd[1,:]))
    
    plt.figure()
    plt.plot(true_density[0], true_density[1], label='true')
    plt.plot(average_ppd[0,:], average_ppd[1,:], linestyle='--', label='estimate')
    plt.xlabel("x", fontsize=18)
    plt.ylabel("density", fontsize=18)
    plt.title('# unsuccesses / total = %d / %d \n Ndata = %d, L1 diff = %.2f' %(unsuccessful_counts,len(estimates), Ndata, L1_diff), fontsize=18)
    # plt.legend(loc='upper left', fontsize=18, bbox_to_anchor=(1.05, 1))
    plt.legend(loc='best', fontsize=18)
    savefigpath = savedirroot + "/coupled_estimates_hType=%s_initType=%s_maxTime=%d_burnin=%d_minIter=%d.png" %(h_type, init_type, max_time,k,m)
    print("Will save figure to %s" %savefigpath)
    # plt.tight_layout()
    plt.savefig(savefigpath)
    plt.show()
    
    return 

## meeting times experiments ------------------------------------------------
def plot_runtimes(savedirroot, h_type, init_type, max_time,k,m):    
    """
    Inputs:
        savedirroot: str, 
        h_type: str
        init_type: str
        max_time, k, m: scalars,
    Outputs:
    
    """
    # load estimates
    estimatespath = savedirroot + "/runtimes_estimates_hType=%s_initType=%s_maxTime=%d_burnin=%d_minIter=%d.pkl" %(h_type, init_type, max_time,k,m)
    print("Loading estimates from %s" %estimatespath)
    estimates = pickle.load(open(estimatespath, "rb"))

    # runtime plot for single-chain. For now just look at one replicate
    single_runtimes = estimates[0][0][0][0] # wow should try to simplify this 
    single_runtimes = np.array(single_runtimes)
    """
    print(len(single_runtimes))
    print(single_runtimes)
    print(np.array(range(1,len(single_runtimes)+1)))
    """
    avg_per_sweep_time = np.mean(single_runtimes/(np.array(range(1,len(single_runtimes)+1))))
    plt.figure(figsize=(8,6))
    plt.plot(single_runtimes, linewidth=3)
    plt.ylabel("Processor time",fontsize=18)
    plt.xlabel("Number of sweeps",fontsize=18)
    plt.title("Cumulative processor time versus single-chain sweeps. \nAverage per-sweep runtime is %.4f" %avg_per_sweep_time,fontsize=18)
    savepath = savedirroot + "/single_runtimes_estimates_hType=%s_initType=%s_maxTime=%d_burnin=%d_minIter=%d.png" %(h_type, init_type, max_time,k,m)
    print("Will save single-chain runtimes to to %s" %savepath)
    plt.savefig(savepath)

    # runtime plots for coupled-chains
    coupled_runtimes = []
    for i in range(int(len(estimates)/2),len(estimates)):
        expers = estimates[i][0]
        for exp in expers:
            coupled_runtimes.append(exp[0])
    num_exp = len(coupled_runtimes)
    print("Number of coupling experiments = %d" %num_exp)

    ## plot 16 coupling experiments
    max_num_plots = 16
    fig, axes = plt.subplots(4,4, figsize=(20,20))
    for idx, ax in zip(range(max_num_plots), axes.flatten()):
        if (idx < num_exp):
            ax.plot(coupled_runtimes[idx], linewidth=3)
            avg_per_sweep_time = np.mean(coupled_runtimes[idx]/(np.array(range(1,len(coupled_runtimes[idx])+1))))
            ax.set_title("Per-sweep time %.2f" %(avg_per_sweep_time),fontsize=18)
        if (idx == 0):
            ax.set_xlabel("Number of sweeps",fontsize=18)
            ax.set_ylabel("Processor time",fontsize=18)
    plt.tight_layout()

    savepath = savedirroot + "/coupled_runtimes_estimates_initType=%s_hType=%s_maxTime=%d_burnin=%d_minIter=%d.png" %(h_type, init_type, max_time,k,m)
    print("Will save coupled-chain runtimes to to %s" %savepath)
    plt.savefig(savepath)
    return

def plot_meetingtimes(savedirroot, h_type, init_type, max_time,k,m):
    # load estimates
    estimatespath = savedirroot + "/runtimes_estimates_hType=%s_initType=%s_maxTime=%d_burnin=%d_minIter=%d.pkl" %(h_type, init_type, max_time,k,m)
    print("Loading estimates from %s" %estimatespath)
    results = pickle.load(open(estimatespath, "rb"))

    # extract distribution of meeting times. Get number of replicates that had no meeting
    coupled_runtimes = []
    unsuccessful_counts = 0
    observed_meetings = []
    censored_meetings = []
    for i in range(int(len(results)/2),len(results)):
        expers = results[i][0] # results[i][1] would be the random seed
        num_success = 0
        for exp in expers: # exp[0] = list of elasped time, exp[1] = meeting time, or None
            tau = exp[1] 
            coupled_runtimes.append(exp[0])
            # didn't meet
            if (tau is None):
                censored_meetings.append(len(exp[0]))
            # did meet
            else:
                num_success += 1
                observed_meetings.append(tau)
        if (num_success == 0):
            unsuccessful_counts += 1

    # plot runtime versus sweep count
    num_exp = len(coupled_runtimes)
    max_num_plots = 16
    fig, axes = plt.subplots(4,4, figsize=(20,20))
    for idx, ax in zip(range(max_num_plots), axes.flatten()):
        if (idx < num_exp):
            ax.plot(coupled_runtimes[idx], linewidth=3)
            avg_per_sweep_time = np.mean(coupled_runtimes[idx]/(np.array(range(1,len(coupled_runtimes[idx])+1))))
            ax.set_title("Per-sweep time %.2f" %(avg_per_sweep_time),fontsize=18)
        if (idx == 0):
            ax.set_xlabel("Number of sweeps",fontsize=18)
            ax.set_ylabel("Processor time",fontsize=18)
    savepath = savedirroot + "/coupled_runtimes_hType=%s_initType=%s_maxTime=%d_burnin=%d_minIter=%d.png" %(h_type, init_type, max_time,k,m)
    print("Will save distribution of runtime versus sweep count to %s" %savepath)
    plt.tight_layout()
    plt.savefig(savepath)

    # distribution of meeting times (potentially censored)
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    axes[0].set_title("Observed meeting times.\nNumber of observations %d" %len(observed_meetings),fontsize=14)
    axes[0].hist(observed_meetings, bins=list(range(0,m+1)))
    axes[0].set_xlabel("Sweep count",fontsize=18)
    axes[0].set_ylabel("Number of observations",fontsize=18)
    axes[1].set_title("Lower bounds of meeting times.\nNumber of observations %d" %len(censored_meetings),fontsize=14)
    axes[1].hist(censored_meetings)
    axes[1].set_xlabel("Sweep count",fontsize=18)
    axes[1].set_ylabel("Number of observations",fontsize=18)
    fig.suptitle("Number of unsuccessful replicates %d / %d" %(unsuccessful_counts, len(results)/2), fontsize=14)
    plt.subplots_adjust(top=0.8) 
    savepath = savedirroot + "/coupled_meetingtimes_hType=%s_initType=%s_maxTime=%d_burnin=%d_minIter=%d.png" %(h_type, init_type, max_time,k,m)
    print("Will save distribution of censored meeting times to %s" %savepath)
    plt.savefig(savepath)

    return
