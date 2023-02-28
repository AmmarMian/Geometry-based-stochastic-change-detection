
import os, shutil
import argparse
from subprocess import run
import stat


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Utility script for launching jobs for change detection on UAVSAR')
    parser.add_argument('scene', metavar='s', type=int, default=1,
                        help='Scene to compute the change detection')
    parser.add_argument('n_repeats', type=int, default=1,
                        help='number of times we repeat time series')
    parser.add_argument('-t','--submit_template',
                        help='HTcondor submit template', required=True) 
    parser.add_argument('-c','--crop_indexes', nargs='+', 
                        help='Cropping indexes of the image', 
                        required=False, default=None) 
    args = parser.parse_args()


    if args.crop_indexes is not None:
        arg_crop = '-'.join(args.crop_indexes)
    else:
        arg_crop = 'None'

    results_dir = f'results/scene{args.scene}_crop{arg_crop}_repeat{args.n_repeats}'
    print(f"Creating results directory: {results_dir}")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    print("Submitting jobs through HTCondor")

    for statistic in ['gaussian_glrt', 'scaled_gaussian_glrt', 'scaled_gaussian_kron_glrt', 'scaled_gaussian_sgd', 'scaled_gaussian_kron_sgd']:
        statistic_result_dir = os.path.join(results_dir, statistic)
        if not os.path.exists(statistic_result_dir):
            os.mkdir(statistic_result_dir)


        # Creating a submit file based upon template
        shutil.copy(args.submit_template, os.path.join(statistic_result_dir, "job.submit"))
        with open(os.path.join(statistic_result_dir, "job.submit"), 'a') as f:
            f.write(f"executable={os.path.join(statistic_result_dir, 'job.sh')}\n")
            f.write(f"log={os.path.join(statistic_result_dir, 'job.log')}\n")
            f.write(f"output={os.path.join(statistic_result_dir, 'job.output')}\n")
            f.write(f"error={os.path.join(statistic_result_dir, 'job.error')}\n")
            f.write(f"queue")

        execute_command = f'python compute_CD_realdata.py {statistic_result_dir} {args.scene} {args.n_repeats} -d {statistic}'
        if args.crop_indexes is not None:
            arg_crop = ' '.join(args.crop_indexes)
            execute_command += f' -c {arg_crop}'
        
        # Creating bash file to run command and tell experiment ended
        with open(os.path.join(statistic_result_dir, "job.sh"), 'w') as f:
            f.write("#!/usr/bin/bash\n")
            f.write(f"cd {os.getcwd()}\n")
            f.write('eval "$(conda shell.bash hook)"\n')
            f.write("conda activate /uds_data/listic/amian/conda/envs/kronecker_online\n")
            f.write(execute_command+"\n")
            f.write(f"echo \"Job Done. Quitting.\"")

        # Make job.sh executable
        st = os.stat(os.path.join(statistic_result_dir, "job.sh"))
        os.chmod(os.path.join(statistic_result_dir, "job.sh"), st.st_mode | stat.S_IEXEC)

        # Submitting for present test statistic
        if args.crop_indexes is not None:
            idcrop = 'Cropped'
        else:
            idcrop = 'Full'
        run(
            [
            "condor_submit" , "-batch-name", f"CD-scene{args.scene}-repeat{args.n_repeats}-{idcrop}", 
            f"{os.path.join(statistic_result_dir, 'job.submit')}"
            ]
        )
        
