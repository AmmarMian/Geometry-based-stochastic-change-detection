# ========================================
# FileName: experiments_scene3.sh
# Date: 13 mars 2023 - 18:20
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: <brief>
# =========================================

# Small crop repeat 5
ARGS=""
SCENE=3
N_REPEATS=5
CROP="-c 0 500 0 500"
for DETECTOR in "gaussian_glrt" "scaled_gaussian_glrt" "scaled_gaussian_sgd" "scaled_gaussian_kron_glrt" "scaled_gaussian_kron_glrt"
do
ARGS="$ARGS --execute_args \"$SCENE $N_REPEATS $CROP -d $DETECTOR\""
done
cmd="python launch_experiment.py experiments/real_data/ $ARGS --runner job --n_cpus 8 --memory 2GB --tag Scene3 --tag repeat_high --tag cropped "
eval $cmd


# All scene repeat 3
ARGS=""
SCENE=3
N_REPEATS=3
CROP=""
for DETECTOR in "gaussian_glrt" "scaled_gaussian_glrt" "scaled_gaussian_sgd" "scaled_gaussian_kron_glrt" "scaled_gaussian_kron_glrt"
do
ARGS="$ARGS --execute_args \"$SCENE $N_REPEATS $CROP -d $DETECTOR\""
done
cmd="python launch_experiment.py experiments/real_data/ $ARGS --runner job --n_cpus 8 --memory 16GB --tag Scene3 --tag repeat_high --tag cropped "
eval $cmd
