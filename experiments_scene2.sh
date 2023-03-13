# ========================================
# FileName: experiments_scene2.sh
# Date: 13 mars 2023 - 18:20
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: <brief>
# =========================================

# Small crop repeat 20
ARGS=""
SCENE=2
N_REPEATS=20
CROP="-c 1 201 1 201"
for DETECTOR in "gaussian_glrt" "scaled_gaussian_glrt" "scaled_gaussian_sgd" "scaled_gaussian_kron_glrt" "scaled_gaussian_kron_glrt"
do
ARGS="$ARGS --execute_args \"$SCENE $N_REPEATS $CROP -d $DETECTOR\""
done
cmd="python launch_experiment.py experiments/real_data/ $ARGS --runner job --n_cpus 8 --memory 2GB --tag Scene2 --tag repeat_high --tag cropped "
eval $cmd


# All scene repeat 10
ARGS=""
SCENE=2
N_REPEATS=10
CROP=""
for DETECTOR in "gaussian_glrt" "scaled_gaussian_glrt" "scaled_gaussian_sgd" "scaled_gaussian_kron_glrt" "scaled_gaussian_kron_glrt"
do
ARGS="$ARGS --execute_args \"$SCENE $N_REPEATS $CROP -d $DETECTOR\""
done
cmd="python launch_experiment.py experiments/real_data/ $ARGS --runner job --n_cpus 8 --memory 16GB --tag Scene2 --tag repeat_high --tag all "
eval $cmd


# All scene repeat 20
ARGS=""
SCENE=2
N_REPEATS=20
CROP=""
for DETECTOR in "gaussian_glrt" "scaled_gaussian_glrt" "scaled_gaussian_sgd" "scaled_gaussian_kron_glrt" "scaled_gaussian_kron_glrt"
do
ARGS="$ARGS --execute_args \"$SCENE $N_REPEATS $CROP -d $DETECTOR\""
done
cmd="python launch_experiment.py experiments/real_data/ $ARGS --runner job --n_cpus 8 --memory 16GB --tag Scene2 --tag repeat_high --tag all"
eval $cmd
