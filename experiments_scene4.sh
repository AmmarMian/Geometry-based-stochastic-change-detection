# ========================================
# FileName: experiments_scene4.sh
# Date: 13 mars 2023 - 18:20
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: <brief>
# =========================================

# Small crop repeat 10
ARGS=""
SCENE=4
N_REPEATS=10
CROP="-c 2800 3000 2800 3000"
for DETECTOR in "gaussian_glrt" "scaled_gaussian_glrt" "scaled_gaussian_sgd" "scaled_gaussian_kron_glrt" "scaled_gaussian_kron_glrt"
do
ARGS="$ARGS --execute_args \"$SCENE $N_REPEATS $CROP -d $DETECTOR\""
done
cmd="python launch_experiment.py experiments/real_data/ $ARGS --runner job --n_cpus 8 --memory 64GB --tag Scene4 --tag repeat_high --tag cropped "
eval $cmd


# Big crop scene repeat 5
ARGS=""
SCENE=4
N_REPEATS=5
CROP="-c 2000 3000 1000 2000"
for DETECTOR in "gaussian_glrt" "scaled_gaussian_glrt" "scaled_gaussian_sgd" "scaled_gaussian_kron_glrt" "scaled_gaussian_kron_glrt"
do
ARGS="$ARGS --execute_args \"$SCENE $N_REPEATS $CROP -d $DETECTOR\""
done
cmd="python launch_experiment.py experiments/real_data/ $ARGS --runner job --n_cpus 8 --memory 128GB --tag Scene4 --tag repeat_high --tag cropped "
eval $cmd
