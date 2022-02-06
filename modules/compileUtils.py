## todos 
## make the compilation functions a class
## unify with compile_mse.py
## add plot to compare number of sweeps done by single and the error in estimation.

import copy
import json
# import statsmodels.api as sm
# import statsmodels 
from scipy import stats
from numpy.random import default_rng
import pickle
import numpy as np
np.set_printoptions(threshold=100)
import pandas as pd
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter
import seaborn as sns

## our code
import names_and_parser
import aggregators
# import sampler_single

LARGE_SAMPLE_SIZE = 10000
LARGER_SAMPLE_SIZE = 200000

def find_representative_dists(distLists, rng=default_rng(42), compareMax=False, lower_bound=50):
    """
    Inputs:
        distsList: list of dists as stored in trace.json files
        each is the distance versus nSweep done in coupled chains
    """
    if (compareMax):
        nSweeps = [len(dists["dists"]["partition"]) for dists in distLists]
        indices = np.argsort(nSweeps)[::-1]
        exp1 = distLists[indices[0]]["dists"]["partition"]
        exp2 = distLists[indices[1]]["dists"]["partition"]
    else:
        long_exps = [dists["dists"]["partition"] for dists in distLists if len(dists) >= lower_bound]
#         print("number of long experiments is", len(long_exps))
        toPlot_ind = rng.choice(len(long_exps), size=2, replace=False)
        exp1 = long_exps[toPlot_ind[0]]
        exp2 = long_exps[toPlot_ind[1]]
    return exp1, exp2

def meetingTimeHelper(df):
    """
    
    Input:
        df: data frame, columns "tau" and "hasMet"
            tau: int, meeting time
            hasMet: boolean, whether we observe meeting event
            nXstep: int, number of steps at the end (either due to meeting or censoring)
    Outputs:
        KM fits of the meeting time (in sampling sweeps and in processor time)
    
    """
    
    censoredTimes = []
    for (idx, val) in enumerate(df["tau"]):
        if val == -1:
            censoredTime = df["nXstep"].iloc[idx]
        else:
            censoredTime = val
        censoredTimes.append(censoredTime)
        
    tauFitter = KaplanMeierFitter()
    tauFitted = tauFitter.fit(censoredTimes, df["hasMet"])
    
    timeFitter = KaplanMeierFitter()
    timeFitted = timeFitter.fit(df["timeTaken"], df["hasMet"])
    return tauFitted, timeFitted

## helpers
def scatter_viz(data_c, ground_truth, ax):
    """
    Use either box plot or artificial scatter plot to visualize
    the distribution of the estimates. 
    """
    width, height = 2.5, 3

    rng = default_rng(50)

    viz_count = 1000
    subset = data_c[:viz_count]

    noise = 5e-2
    n_obs = len(subset)
    y = rng.normal(0, scale=noise, size=n_obs)
    aug_data_c = np.vstack((subset, y)).T

    alpha = 0.01
    lo_q = np.quantile(subset, alpha)
    hi_q  = np.quantile(subset, 1-alpha)
    sd = np.std(subset)
    me = np.mean(subset)

    ylim = [-4*noise, 4*noise]
    if (ax is None):
        fig = plt.figure(figsize=(width, height))
        plt.ylim(ylim)
        plt.scatter(aug_data_c[:,0], aug_data_c[:,1], color='blue', s = 6)
        plt.axvline(x=lo_q, linestyle='--', color='red', label='%.2f Quantile' %alpha)
        plt.axvline(x=hi_q, linestyle='-.', color='red', label='%.2f Quantile' %(1-alpha))
        plt.axvline(x = ground_truth, color='black', label='Ground Truth')
        plt.xlabel('Estimates')
        plt.ylabel('Artificial')
        plt.legend(fontsize=8.5, loc='upper right')
        plt.show()
    else:
        ax.set_ylim(ylim)
        ax.scatter(aug_data_c[:,0], aug_data_c[:,1], color='blue', s = 6)
        ax.axvline(x=lo_q, linestyle='--', color='red', label='%.2f Quantile' %alpha)
        ax.axvline(x=hi_q, linestyle='-.', color='red', label='%.2f Quantile' %(1-alpha))
        ax.axvline(x = ground_truth, color='black', label='Ground Truth')
        ax.set_xlabel('Coupled-Chain\nEstimates')
        ax.set_ylabel('Artificial')
#         ax.tick_params(axis='x', labelsize=12)
#         ax.tick_params(axis='y', labelsize=12)
#         ax.legend(fontsize=9, bbox_to_anchor=(0.5, 1.05))
#         ax.legend(fontsize=8)
#         ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
#           ncol=2, fontsize=8)
    return 

def hist_viz(data_s, ground_truth, ax):
    width, height = 2.5, 3
    rng = default_rng(50)
    
    est_mean = np.mean(data_s)

    viz_count = len(data_s)
    subset = data_s[:viz_count]

    alpha = 0.01
    lo_q = np.quantile(subset, alpha)
    hi_q  = np.quantile(subset, 1-alpha)
    sd = np.std(subset)
    me = np.mean(subset)

    if (ax is None):
        fig = plt.figure(figsize=(width, height))
        # xlim = [0.25, 0.6]
        # plt.xlim(xlim)
        # bins = np.arange(0.3, 0.65, step=0.01)
        # print(bins)
        # plt.hist(subset, density=True, bins = bins, color='blue')
        plt.hist(subset, density=True, color='blue')
        plt.axvline(x = ground_truth, color='black', label='Ground Truth')
        plt.axvline(x = est_mean, color='red', label = 'Population Mean')
        plt.xlabel('Estimates')
        plt.ylabel('Density')
        plt.legend(fontsize=10, loc='upper right')
        plt.show()
    else:
        ax.hist(subset, density=True, color='blue', alpha=0.5)
        ax.axvline(x = ground_truth, color='black', label='Ground Truth')
        ax.axvline(x = est_mean, color='red', label = 'Population Mean')
        ax.set_xlabel('Naive Parallel\nEstimates')
        ax.set_ylabel('Density')
#         ax.tick_params(axis='x', labelsize=12)
#         ax.tick_params(axis='y', labelsize=12)
#         ax.legend(fontsize=9, bbox_to_anchor=(0.2, 1.05))
#         ax.legend(fontsize=8)
    return 

# def make_q_list(nProc_list, maxTrim=15):
#     start_q = 0.01
#     q_list = [min(start_q, (maxTrim/2)/nProc) for nProc in nProc_list]
#     return q_list

# def make_q_list(nProc_list):
#     start_q = 0.005
#     q_list = [start_q for nProc in nProc_list]
#     return q_list

def computeErrorHelper(data, trim, ground_truth, nProc_list, qVals, getPerc=True):
    """
    Inputs:
        data: array_like
        trim: scalar, total amount of trimming on left and right tail
        ground_truth: scalar
        nProc_list: array_like, number of processes to batch up before computing estimates
        qVals: dictionary, quantiles to summarize the distribution of losses
        
    Outputs:
        losses: dict,
    """
    
    totRep = len(data)
    
    aggFuncs = {"Trimmed": lambda batch: aggregators.trim_mean(batch, trim/2, trim/2, None, None),
                "Sample": lambda batch: aggregators.arith_mean(batch, None, None)}
    
    def localHelper(func):
        losses = {}
        losses["nProcList"] = nProc_list
        for pIdx, nProc in enumerate(nProc_list):
            indices = range(totRep)[0:totRep:nProc] 
            ests = []

            for idx in indices:
                batch = data[idx:(idx+nProc)] # this batching is consistent across single and coupled
                ests.append(func(batch))

            squaredLoss = np.square(np.array(ests)-ground_truth)

            # compute RMSE and quantiles of the loss distributions
            for key, val in qVals.items():
                if key not in losses:
                    losses[key] = []
                    
                summaryVal = np.sqrt(np.mean(squaredLoss)) if key == "mean" else np.sqrt(np.quantile(squaredLoss, val))
                summaryVal *= 100/ground_truth if getPerc else 1
                losses[key].append(summaryVal)
                    
        return losses
    
    losses = {"Trimmed": localHelper(aggFuncs["Trimmed"]),  "Sample": localHelper(aggFuncs["Sample"])}
    return losses

def get_mean_sem(data, nProc_list):
    rng = default_rng(42)
    means = []
    sems = []
    N = len(data)
    for nProc in nProc_list:
        idx = rng.choice(N, size=nProc, replace=False)
        means.append(np.mean(data[idx]))
        sems.append(stats.sem(data[idx]))
    return np.asarray(means), np.asarray(sems) 

def nSweep_vs_timeTaken(single_aux_info):
    width, height = 6,2.5
    fig, axes = plt.subplots(1,2,figsize=(width, height))
    axes[0].violinplot(single_aux_info["nSweeps"])
    axes[0].set_ylabel('nSweeps')

    subset_size = 1000

    axes[1].scatter(x=single_aux_info["nSweeps"][:subset_size], y=single_aux_info["tTaken"][:subset_size], 
                    color='red', marker='x')
    axes[1].set_xlabel('nSweeps')
    axes[1].set_ylabel('tTaken')

    plt.tight_layout()
    plt.show()
    return 

def nSweep_vs_err(data_s, ground_truth, single_aux_info):
    width, height = 3,1.5
    subset_size = 1000
    err = 100*(data_s[:subset_size] - ground_truth)/ground_truth
    nSweeps = single_aux_info["nSweeps"][:subset_size]
    plt.figure(figsize=(width, height))
    plt.scatter(x=nSweeps, y=err, color='red', marker='x')
    plt.xlabel('nSweeps')
    plt.ylabel('Percentage Error')
    plt.show()
    return 

class FigureMaker():
    def __init__(self, stylefile, dirArgs, funcArgs, estArgs):
        
        """
        Inputs:
            estDir: str, directory with groundTruth.csv, estType=coupled.csv, estType=single.csv
            meetingDir: str, directory with 
        """

        plt.style.use(stylefile)

        targetIdx = self.loadTruth(dirArgs["estDir"], funcArgs)
        self.loadEstimates(dirArgs["estDir"], targetIdx)
        print('===========================')
        if dirArgs["meetingDir"] is not None:
            self.makeSurvivalCurves(dirArgs["meetingDir"], dirArgs["subDirs"])
            if (dirArgs["getTraces"]):
                self.getDistTraces(dirArgs["meetingDir"], dirArgs["subDirs"])
        print('===========================')
        self.computeErrors(estArgs["trim"], estArgs["ErrorNProcList"])
        print('===========================')
        self.computeIntervals(estArgs["IntervalNProcList"])
        return 
    
    def loadTruth(self, estDir, funcArgs):
        
        groundTruthDf = pd.read_csv(estDir + "/groundTruth.csv")
        if funcArgs["hType"] == "LCP":
            targetIdx = funcArgs["topKIdx"] # target function = proportion of cluster ranked by size
            groundTruthRow = groundTruthDf.iloc[targetIdx,:]
        else:
            assert funcArgs["hType"] == "CC"
            str_ = funcArgs["CCPair"].split(",")
            target_pair = int(str_[0]), int(str_[1])
            CCIndices = funcArgs["CCIndices"] # need to match the CCIndices in json files
            nObs = int(np.sqrt(groundTruthDf.shape[0]))
            pairInCCIndices = (CCIndices[target_pair[0]], CCIndices[target_pair[1]])
            targetIdx = np.ravel_multi_index(pairInCCIndices, [nObs, nObs])
            groundTruthRow = groundTruthDf.iloc[targetIdx,:]
        self.groundTruth = groundTruthRow["groundTruth"]
         
        return targetIdx
    
    def loadEstimates(self, estDir, targetIdx):
        singleDf = pd.read_csv(estDir + "/estType=single.csv")
        self.single = singleDf.filter(regex=("est*")).iloc[:,targetIdx]
        
        print("descriptive stats for naive parallel estimates")
        print("\t",stats.describe(self.single))
        print("check if error is within CLT predictions")
        relErr = (np.mean(self.single)-self.groundTruth)/stats.sem(self.single)
        print("\tdifference between naive parallel estimates' mean and ground_truth divided by standard error = %.2f\n" %relErr)
        
        coupledDf = pd.read_csv(estDir + "/estType=coupled.csv")
        self.coupled = coupledDf.filter(regex=("est*")).iloc[:,targetIdx]
        
        print("descriptive stats for coupled estimates")
        print("\t",stats.describe(self.coupled))
        print("check if error is within CLT predictions")
        relErr = (np.mean(self.coupled)-self.groundTruth)/stats.sem(self.coupled)
        print("\tdifference between coupled estimates' mean and ground_truth divided by standard error = %.2f\n" %relErr)
        
        return 
    
    def computeErrors(self, trim, nProcList):
        """
        Inputs:
            trim: scalar,
            nProcList: list,
            
        Outputs:
        """
        
        print("batching up estimates and computing errors")
        
        
        qVals = {"mean": None, "low": 0.2, "medium": 0.5, "high": 0.8}
        
        errors = {}
        errors["Coupled"] = computeErrorHelper(self.coupled, trim, self.groundTruth, nProcList, qVals)
        errors["NP"] = computeErrorHelper(self.single, trim, self.groundTruth, nProcList, qVals)
        
        self.errors = errors
        return
    
    def computeIntervals(self, nProcList):
        print("constructing confidence intervals based on batched estimates")
        
        cMean, cSe = get_mean_sem(self.coupled, nProcList)
        sMean, sSe = get_mean_sem(self.single, nProcList)
        self.intervals = {"nProcList": nProcList, "cMean": cMean, "cSe": cSe, "sMean":sMean, "sSe": sSe}
        return 
        
    def makeSurvivalCurves(self, meetingDir, couplingDict):
        
        """
        Inputs:
            meetingDir: str
            couplingDict: dict, key, val pairs have the form "OT":("OTHamming/", "blue")
                    where the first tuple element is the subdirectory and the second is the color
                    used to plot
        """
        print("loading meeting time data")
        
        curves = {}
#         subDirs = {"CommonRNG": "CommonRNG/", "Maximal": "Maximal/", "OT":"OTHamming/", "OT-VI":"OTVI/"}
        
        for key, val in couplingDict.items():
            path = meetingDir + val[0] + "/estType=coupled.csv"
            df = pd.read_csv(path)
            tauFitted, timeFitted = meetingTimeHelper(df)
            curves[key] = (tauFitted, timeFitted, val[1])
        self.curves = curves
        return 
    
    def getDistTraces(self, meetingDir, couplingDict):
        print("loading representative distance traces")
        
        lower_bound = 0
        
        repTraces = {}
#         subDirs = {"CommonRNG": "CommonRNG/", "Maximal": "Maximal/", "OT":"OTHamming/", "OT-VI":"OTVI/"}
        
        for key, val in couplingDict.items():
            path = meetingDir + val[0] + "/traces.json"
            traces = json.load(open(path, "r"))
            exp1, exp2 = find_representative_dists(traces, lower_bound=lower_bound)
            repTraces[key] = (exp1, exp2, val[1])
        self.repTraces = repTraces
        
        return 
    
    def visualizeEstimates(self):
        print("\n----------------------------------------")
        print("visualize estimates")

        fig, axes = plt.subplots(1,2, figsize=(5, 3))
        print("scatter plot of coupled estimates better legibility")
        scatter_viz(self.coupled, self.groundTruth, axes[0])

        print("\n----------------------------------------")
        print("histogram of single estimates")
        hist_viz(self.single, self.groundTruth, axes[1])
        plt.tight_layout()
        plt.show()
        return 
    
    def plotSurvivalCurves(self, ax, drawArgs, tau=False):
        
        xlabel = 'Meeting Time (sweeps)' if tau else 'Meeting Time (seconds)'
        
        for key, val in self.curves.items():
            tauCurve, timeCurve, color = self.curves[key]
            # left panel plots the survival functions
            if tau:
                tauCurve.plot_survival_function(ax=ax, color=color, label=key, legend=None)
            # right panel plots the distance traces
            else:
                timeCurve.plot_survival_function(ax=ax, color=color, label=key, legend=None)
                
#         ax.set_xscale('log')
        ax.set_yscale('log')

#       ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
        if "drawLabel" in drawArgs:
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Survival Function')
            
        if "legend" in drawArgs:
            ax.legend(fontsize=8)
            
        if "title" in drawArgs:
            ax.set_title("Coupling Choice")
            
#         axes[1].set_xlabel('Number of Sweeps')
#         axes[1].set_ylabel('Distance')
#         axes[1].set_yscale('log')
#         plt.tight_layout()
#         plt.show()
        return 
    
    
    def plotMainFigureRow(self, errorYLim, firstRow, **kwargs):
        
        fig, axes = plt.subplots(1,4, **kwargs)
        
        fullDrawArgs = []
        reducedDrawArgs = []
        if firstRow:
            fullDrawArgs = ["drawLabel", "legend", "title"]
            reducedDrawArgs = ["drawLabel", "title"]
        
        self.plotErrors(axes[0], "Sample", errorYLim, fullDrawArgs)
        self.plotErrors(axes[1], "Trimmed", errorYLim, reducedDrawArgs)
        
        self.plotIntervals(axes[2], reducedDrawArgs)
        self.plotSurvivalCurves(axes[3], fullDrawArgs)
        
        plt.show()
        
        return 
    
    def plotErrors(self, ax, aggType, errorYLim, drawArgs):
        
        ax.set_ylim(errorYLim)
        
        for estType in ["Coupled", "NP"]:
            color = "blue" if estType == "Coupled" else "red"
            data = self.errors[estType][aggType]
            ax.plot(data["nProcList"], data["medium"], marker='x', color=color, label=estType)
            ax.fill_between(data["nProcList"], data["low"], data["high"], color=color, alpha=0.3)
            
        if "drawLabel" in drawArgs:
            ax.set_xlabel('Number of Processes')
            ax.set_ylabel('Percentage Error')
            
        if "legend" in drawArgs:
            ax.legend(fontsize=8)
            
        if "title" in drawArgs:
            ax.set_title("%s Mean" %aggType)
            
        return 
    
    def plotIntervals(self, ax, drawArgs):
        
    #     axes[0].legend(fontsize=8, loc='upper right', bbox_to_anchor=(1, 1.2))
    #     axes[0].legend(fontsize=8)

        
        ax.errorbar(x=self.intervals["nProcList"], y=self.intervals["cMean"], color='blue', yerr=2*self.intervals["cSe"], label='Coupled')
        ax.errorbar(x=self.intervals["nProcList"], y=self.intervals["sMean"], color='red', yerr=2*self.intervals["sSe"], label='NP')
        ax.axhline(y=self.groundTruth, color='black', linestyle='--', label='Ground Truth')
        
        if "drawLabel" in drawArgs:
            ax.set_xlabel('Number of Processes')
            ax.set_ylabel('Estimate (+- 2SEM)')
            
        if "legend" in drawArgs:
            ax.legend(fontsize=8)
            
        if "title" in drawArgs:
            ax.set_title("Intervals")
        # print(len(jobs_per_task), np.unique(jobs_per_task))
        # axes[1].legend(fontsize=8)
#         plt.tight_layout()
#         plt.show()
        
        return 
    
    def plotRMSE(self, ax, fontsize):
        
#         fig, axes = plt.subplots(1,2,**kwargs)
#         axes[0].set_xlabel('Number of Processes')
#         axes[0].set_ylabel('Percentage Error')
        
#         errors = self.computeErrors(trim, RMSENProcList)
        for estType in ["Coupled", "NP"]:
            color = 'blue' if estType == "Coupled" else 'red'
            for aggType in ["Trimmed", "Sample"]:
                linestyle = '-' if aggType == "Sample" else '--'
                data = self.errors[estType][aggType]['mean']
#                 print(data)
                ax.plot(data["nProcList"], data,  marker='x',color=color, linestyle=linestyle , label='%s, %s' %(estType, aggType))
        
        return 
    
    def plotDistTraces(self, ax, drawArgs):
        
        legend_elements = []
        
        for key, val in self.repTraces.items():
            exp1, exp2, color = val
            ax.plot(exp1, color=color)
            ax.plot(exp2, color=color)
            legend_elements.append(Line2D([0], [0], color=color, lw=3, label=key))
        
        ax.set_yscale('log')

#       ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
        if "drawLabel" in drawArgs:
            ax.set_xlabel('Number of Sweeps')
            ax.set_ylabel('Distance')
            
        if "legend" in drawArgs:
            ax.legend(handles=legend_elements, fontsize=8)
            
        return 
    
if __name__ == "__main__":
    options = names_and_parser.parse_args()
    if (options.compileType == "individual"):
        compileMSEVsNProc(options)

    elif (options.compileType == "across_settings"):
        vary_options = names_and_parser.parse_compile_file_type2(options.compile_settings_file)
        compare_meeting_across_settings(options, vary_options)

