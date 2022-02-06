## todos
## vectorize arith_mean and trimmed_mean

import numpy as np
from numpy.random import default_rng
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('aistats2022')

## ----------------------------------------------------------------------
## given Monte Carlo samples, how to combine them?

## median-of-means
def med_of_m(x, K, BT_size=None, true_val=None):
    """
    Inputs:
        x: (N,) array, realizations of some r.v.
        K: scalar, number of groups to compute means
        BT_size: scalar or None, number of bootstrap samples
        true_val: scalar or None, reference point to compute RMSE
    Outputs:
        est: scalar, point estimate based on original sample
        if BT_size not None
            sd: scalar, standard deviation based on BT_size bootstrap samples
            rmse: scalar or None
            bt_est: (BT_size,), bootstrap samples
    """
    N = len(x)
    assert N % K == 0
    B = N // K
    
    means = [np.mean(x[i*B:(i+1)*B]) for i in range(K)]
    est = np.median(means)
    
    if not (BT_size is None): 
        rng = default_rng(42)
        samples = rng.choice(x, size=(BT_size, N)) # each bootstrap sample is in a row

        bt_means = np.array([np.mean(samples[:,i*B:(i+1)*B], axis=1) for i in range(K)])
        bt_est = np.median(bt_means, axis=0)
        sd = np.std(bt_est)
        if (true_val is None):
            rmse = None
        else:
            rmse = np.sqrt(np.mean(np.square(bt_est-true_val)))
        return est, sd, rmse, bt_est
    else:
        return est

## simple arithmetic mean
def arith_mean(x, BT_size=None, true_val=None):
    """
    Inputs:
        x: (N,) array, realizations of some r.v.
        BT_size: scalar (or None), number of bootstrap samples
        true_val: scalar, reference point to compute RMSE
    Outputs:
        est: scalar, point estimate based on original sample
        if BT_size not None
            sd: scalar, standard deviation based on BT_size bootstrap samples
            rmse: scalar, bootstrap estimate of RMSE
            bt_est: (BT_size,), bootstrap samples
    """
    N = len(x)
    est = np.mean(x)
    
    if not (BT_size is None): 
        rng = default_rng(42)
        samples = rng.choice(x, size=(BT_size, N)) # each bootstrap sample is in a row
        bt_est = np.mean(samples, axis=1)
        sd = np.std(bt_est)
        if (true_val is None):
            rmse = None
        else:
            rmse = np.sqrt(np.mean(np.square(bt_est-true_val)))
        return est, sd, rmse, bt_est
    else:
        return est
    
## trimmed mean 
def trim_mean(x, left_q, right_q, BT_size=None, true_val=None):
    """
    Report data descriptive statistics after trimming 100q th and 100 - 100q quantiles
    from data.
    
    Inputs:
        x: (n,) array
        left_q: scalar, valued in (0,0.5), typically q = 0.005
        right_q: scalar, valued in (0,0.5), typically q = 0.005
        BT_size: scalar (or None), number of bootstrap samples
        true_val: scalar, reference point to compute RMSE
    Outputs:
        est: scalar, point estimate based on original sample
        if BT_size not None
            sd: scalar, standard deviation based on BT_size bootstrap samples
            rmse: scalar, bootstrap estimate of RMSE
            bt_est: (BT_size,), bootstrap samples
    """
    assert left_q < 0.5 and right_q < 0.5
    
    N = len(x)
    lo_q = np.quantile(x, left_q)
    hi_q = np.quantile(x, 1-right_q)
    trimmed_data = x[(x > lo_q) & (x < hi_q)]
    est = np.mean(trimmed_data)
    
    if not BT_size is None:
        # get bootstrap distribution to estimate sd and rmse
        rng = default_rng(42)
        samples = rng.choice(x, size=(BT_size, N)) # each bootstrap sample is in a row

        lo_q = np.quantile(samples, q, axis=1)
        hi_q = np.quantile(samples, 1-q, axis=1)

        # for loop is a start but should be improved
        trimmed_data = []
        bt_est = []
        for i in range(BT_size):
            data = samples[i,:]
            trimmed_data = data[(data > lo_q[i]) & (data < hi_q[i])]
            curr = np.mean(trimmed_data)
            bt_est.append(curr)
        sd = np.std(bt_est)
        if (true_val is None):
            rmse = None
        else:
            rmse = np.sqrt(np.mean(np.square(bt_est-true_val)))
        return est, sd, rmse, bt_est
    else:
        return est

def synthetic(mu, p, N, B, display=True):
    # Inputs:
        #  N: scalar, number of draws
        # mu: scalar, mean of right outlier component 
        # p: scalar, proportion of central component (close to 1)
        # B: scalar, number of Monte Carlo draws
    # Outputs:

    # AISTATS specs
    width, height = 1.5, 1.5
    legend_fs = 8

    rng = default_rng(42)

    lp, cp, rp = 0.5-p/2, p, 0.5-p/2

    def sample(a):
        comps = rng.choice(3, p=[lp, cp, rp], size=(a,))
        x = np.zeros(a)
        for i in range(a):
            if (comps[i] == 0):
                x[i] = rng.normal(-mu, 1)
            elif (comps[i] == 1):
                x[i] = rng.normal(0, 1)
            elif (comps[i] == 2):
                x[i] = rng.normal(mu, 1)
        return x

    x = sample(B*N)
    if (display):
        print("histogram of distribution")
        plt.figure()
        bins = np.arange(-3*mu, 3*mu, mu/10)
        plt.hist(x, bins=bins, density=True)
        plt.show()
        print("\n----------------------------------------\n")

    x = np.reshape(x, (B,N))
    smeans = np.mean(x, axis=1)
    q = 1.2*(0.5-p/2) # trimming amount slightly larger than component mass
    tmeans = np.zeros(B)
    for i in range(B):
        tmeans[i] = trim_mean(x[i,:], q)

    if (display):
        df = pd.DataFrame(list(zip(tmeans, smeans)), columns=['trimmed', 'sample'])
        plt.figure(figsize=(width, height))
        sns.boxplot(data=df, orient='h')
        plt.show()

        print("\n----------------------------------------\n")
    smean_rmse = np.sqrt(np.mean(np.square(smeans)))
    tmean_rmse = np.sqrt(np.mean(np.square(tmeans)))

    if (display):
        print("Sample mean rmse", smean_rmse)
        print("Trimmed mean rmse", tmean_rmse)

    return tmeans, smeans, smean_rmse, tmean_rmse 

if __name__ == "__main__":
    
    Nlist = [250, 500, 750, 1000, 1250]
    B = 400
    mu = 7
    p = 0.9
    smean_rmses, tmean_rmses = np.zeros(len(Nlist)), np.zeros(len(Nlist))
    for idx, N in enumerate(Nlist):
        _, _, smean_rmse, tmean_rmse = synthetic(mu, p, N, B, False)
        smean_rmses[idx] = smean_rmse
        tmean_rmses[idx] = tmean_rmse
        
    N = 1000
    B = 400
    mu = 7
    p = 0.9
    tmeans, smeans, _, _ = synthetic(mu, p, N, B, False)

    width, height = 6, 3
    fig, axes = plt.subplots(1,2,figsize=(width, height))
    axes[0].set_xlabel('Number of Processors')
    axes[0].set_ylabel('RMSE')
    axes[0].plot(Nlist, smean_rmses,  color='blue', marker='x', label='sample')
    axes[0].plot(Nlist, tmean_rmses, color='red', marker='x', label='trimmed')
    axes[0].legend(fontsize=8)
    
    my_pal = {"trimmed": "red", "sample":"blue"}
    df = pd.DataFrame(list(zip(tmeans, smeans)), columns=['trimmed', 'sample'])
    sns.boxplot(data=df, orient='h', ax=axes[1], palette=my_pal)
    plt.tight_layout()
    plt.show()
    
#     N = 1000
#     B = 400
#     mu = 7
#     p = 0.9
#     smean_rmse, tmean_rmse = synthetic(mu, p, N, B, True)
        
