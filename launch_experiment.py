# ========================================
# FileName: launch_experiment.py
# Date: 10 mars 2023 - 10:40
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Utility script to launch an
# experiment, locally or through HTCondor
# =========================================

import os
import sys
import argparse
import htcondor
from rich.console import Console
from tinydb import TinyDB, Query
import datetime
from subprocess import run
import git


def prompt_create_dir(dir_path, console, status):
    console.log(f'Directory {dir_path} does not exist.')
    status.stop()
    choice = console.input('Should I create it for you ? (y/N)')
    if choice == '' or 'y':
        os.mkdir(dir_path)
        status.start()
    else:
        console.log('Exitting...')
        sys.Exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                description="Helper script to manage experiments "
                "for the project of online change detection with "
                "structure"
            )
    parser.add_argument('experiment_folder', help="Experiment folder path. "
                        "Must contains at least one script named execute.sh "
                        "performing the experiment.\n"
                        "It can take command line arguments"
                        "as an input to handle varying hyperparameters. "
                        "Those are specified thanks to the optional "
                        "argument --execute_args\n"
                        "The results directory for this experiment will also"
                        "be passed as the last argument to handle data saving."
                        "Additionally, any number of scripts can be added to "
                        "perform actions after the experiment ended using the "
                        "syntax action_*.sh. in the name of the script."
                        )
    parser.add_argument('runner', choices=['local', 'job'], default='local',
                        help="Specifies how to execute the experiment.\n "
                        "If job is specified. It is possible to specify its "
                        "options using args --n_cpus and --memory."
                        )
    parser.add_argument('--results_dir', type=str, default='results/',
                        help="Directory where experiment results are stored."
                        " Default to ./results/."
                        )
    parser.add_argument('--execute_args', type=str,
                        help="Arguments to pass to "
                        "execute.sh script. MUST BE USED WITH \"\" around.\n"
                        "For example:\n python launch_experiment.py "
                        "experiments/test --execute_args \"--arg1 arg1\"",
                        default=''
                        )
    parser.add_argument('--n_cpus', type=int, default='1',
                        help="Number of cpus to ask for a job."
                        )
    parser.add_argument('--memory', type=str, default='8GB',
                        help="Memory to ask for a job."
                        )
    parser.add_argument('--is_flash', action='store_true', default=False,
                        help='Whether the job is short (Less than hour to have priority scheduling.')
    args = parser.parse_args()

    # Handling terminal info output using rich console
    console = Console()
    with console.status(
            f"[bold]Experiment {args.experiment_folder}: Setting things up"
            ) as status:

        repo = git.Repo('.')
        try:
            assert not repo.is_dirty()
        except AssertionError:
            console.log('[bold red]Repertory is not committed.')
            console.log('Please commit changes before launching an experiment')

        # Handling experiment results directory
        if not os.path.exists(args.results_dir):
            prompt_create_dir(args.results_dir, console, status)

        experiment_id = len([x[1] for x in os.walk(args.results_dir)])
        experiment_basename = os.path.basename(
                os.path.normpath(args.experiment_folder))
        experiment_results_dir = os.path.join(
                args.results_dir, f'{experiment_id}_{experiment_basename}'
                )
        console.log(
                'Creating experiment results directory: '
                f'{experiment_results_dir}'
                )
        os.mkdir(experiment_results_dir)

        # Handling experiment directory
        execute_path = os.path.join(args.experiment_folder, 'execute.sh')
        if not os.path.exists(execute_path):
            console.log(f'[bold red]Execute script {execute_path} not found!')
            console.log('Quitting...')
            sys.exit(0)

        # Experiment database handling
        dB_path = os.path.join(args.results_dir, 'experiments.json')
        console.log(f'Database of experiments stored in: {dB_path}')
        dB = TinyDB(dB_path)

        console.log('Adding experiment to database')
        dB.insert({
                'id': experiment_id,
                'experiment_folder': args.experiment_folder,
                'status': 'not started',
                'arguments': args.execute_args,
                'launch_date': datetime.datetime.now().isoformat(),
                'experiment_results_dir': experiment_results_dir
                })

        # Launching experiment
        if args.runner == "local":
            console.log('Launching from local runner')
            execute_string = f'bash {execute_path} {args.execute_args} {experiment_results_dir}'
            console.log('Executing command:')
            console.log(execute_string+'\n')
            status.update(f"[bold]Experiment {args.experiment_folder}: Running locally"
                    )
            f_stdout = open(os.path.join(experiment_results_dir, 'output.txt'), 'w')
            f_stderr = open(os.path.join(experiment_results_dir, 'error.txt'), 'w')
            dB.update({"status": "running"}, Query().id == experiment_id)
            try:
                run([execute_path, args.execute_args, experiment_results_dir],
                    stdout=f_stdout, stderr=f_stderr)
            except Exception as e:
                console.log('[bold red]Something went wrong. Check log files!')
                console.log(e)
            f_stdout.close()
            f_stderr.close()
            console.log('Experiment done.')
            dB.update({"status": "finished"}, Query().id == experiment_id)
        else:
            console.log('Submitting job to HTCondor')
            job = htcondor.Submit({
                'executable': execute_path,
                'arguments': args.execute_args + f' {experiment_results_dir}',
                'output': os.path.join(experiment_results_dir, 'output.txt'),
                'error': os.path.join(experiment_results_dir, 'error.txt'),
                'log': os.path.join(experiment_results_dir, 'log.txt'),
                'request_cpus': args.n_cpus,
                'request_memory': args.memory,
                'getenv': True,
                'should_transfer_files': 'IF_NEEDED',
                'when_to_transfer_output': 'ON_EXIT',
                '+WishedAcctGroup': "group_usmb.listic",
                '+isFlash': args.is_flash
                })
            console.log(job)
            schedd = htcondor.Schedd() # get the Python representation of the scheduler
            submit_result = schedd.submit(job) # submit the job
            console.log(f'Job sumitted to Cluster {submit_result.cluster()}') # print the job's ClusterId
            submit_info = dict(job)
            submit_info['cluster'] = submit_result.cluster()
            dB.update({
                "status": "running",
                "submit_info": submit_info 
                }, Query().id == experiment_id)
