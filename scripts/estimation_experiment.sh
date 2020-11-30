# estimation_experiment.sh runs all Markov chains for the estimation experiment
# which generated figure 2B
module load anaconda/2020a 
source activate crp
# Model parameters
sd0=0.5
sd=1.3
alpha=1.0

# Long chains for ground truth 
echo "starting long chain for ground truth"
python ../modules/estimation_experiment.py --est_type truth --max_iter 10000 --k 5000 \
    --pool_size 10 --sd0 $sd0  --sd $sd --alpha $alpha > ../logs/estimation_experiment_truth.log

# parallel estimates
max_time=90
pool_size=70
n_replicates=200
k=1
m=120
echo "starting coupled chain ests"
python ../modules/estimation_experiment.py --est_type coupled --k $k --m $m \
    --pool_size $pool_size --n_replicates $n_replicates --max_time $max_time  \
    --sd0 $sd0  --sd $sd --alpha $alpha  > ../logs/estimation_experiment_coupled.log
#
echo "starting single chain ests"
python ../modules/estimation_experiment.py --est_type single --k $k --m $m \
    --pool_size $pool_size --n_replicates $n_replicates --max_time $max_time  \
    --sd0 $sd0  --sd $sd --alpha $alpha  > ../logs/estimation_experiment_single.log
