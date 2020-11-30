# coupling-Gibbs-partition

Environment.yml
- if we load anaconda environment OTcoupling and run "jupyter notebook" from the command line then the Jupyter Notebook can ``import ot''. But once in the Notebook, if we change kernels to ``OTcoupling'', we can't ``import OT''. 

Figure 1 notebook

- run_rep() calls np.random.seed(), so we technically won't be able to exactly replicate the meeting time plots
- takes around 40 minutes to run on Stan's computer, with max_iter = 100


## Figure 2B
We have also included the data and code we used to generate figure 2B so that anyone interested may view the details of our analysis.
Howeve, because of the large computational resources demanded by this experiment
and the differences we expect amongst the compute platforms that will be available 
to those interested, we have not intended to make this analysis replicable on the press of a button (as with figure 1).
For our experiments, we used [Supercloud](https://supercloud.mit.edu/) [[1]](#1).
Instead we include the key pieces of our pipeline.
* [gene\_data\_Ndata=200\_D=50.csv](data/gene_data_Ndata=200_D=50.csv) contains the processed single-cell RNA-seq dataset.  This was initially published in [[2]](#2) and processed and modeled using a Dirichlet process mixture model (our starting point) by [[3]](#3).
* [estimation\_experiment.sh](scripts/estimation_experiment.sh) is the script we used to run our parallel estimation experiments. This runs batches of parallel chains in sequence to simulate higher parallelism.
* [compile\_results.py](modules/compile_results.py) contains helper functions we used to qualitatively evaluate the procedure while exploring its sensitivities to parameters on our own.
* [Figure2.ipynb](Figure2.ipynb) plots a processed set of results extracted
  from the output of [estimation\_experiment.sh](scripts/estimation_experiment.sh) to generate the panel included in the paper.


## References
<a id="1"> [1] </a>
Interactive Supercomputing on 40,000 Cores for Machine Learning and Data Analysis, Albert Reuther, Jeremy Kepner, Chansup Byun, Siddharth Samsi, William Arcand, David Bestor, Bill Bergeron, Vijay Gadepally, Michael Houle, Matthew Hubbell, Michael Jones, Anna Klein, Lauren Milechin, Julia Mullen, Andrew Prout, Antonio Rosa, Charles Yee, Peter Michaleas, paper presented at the 2018 IEEE High Performance Extreme Computing Conference (HPEC), July 2018.

<a id="2"> [2] </a>  Amit Zeisel, Ana B Mun ̃oz-Manchado, Simone Codeluppi, Peter L ̈onnerberg, Gioele La Manno, Anna Jur ́eus, Sueli Marques, Hermany Munguba, Liqun He, Christer Betsholtz, et al. Cell types in the mouse cortex and hippocampus revealed by single-cell RNA-seq. Science, 347(6226):1138–1142, 2015.

<a id="3"> [3] </a>
Sandhya Prabhakaran, Elham Azizi, Ambrose Carr, and Dana Peer. Dirichlet process mixture model for correcting technical variation in single-cell gene expression data. In International Conference on Machine Learning, pages 1070–1079, 2016.
