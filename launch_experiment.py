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
from rich import print as rprint
from rich.console import Console
from tinydb import TinyDB, Query
import datetime
from subprocess import run
import git
import copy
import time
import pathlib
try:
    import htcondor
    htcondor_available = True
except ModuleNotFoundError:
    rprint(
        "[bold green]HTcondor module not found, job runner is unaivalaible.")
    htcondor_available = False

def prompt_create_dir(dir_path, console, status):
    console.log(f'Directory {dir_path} does not exist.')
    status.stop()
    choice = console.input('Should I create it for you ? (y/N) ')
    if choice == '' or 'y':
        os.mkdir(dir_path)
        status.start()
    else:
        console.log('Exitting...')
        sys.Exit(0)


def execute_locally(console, status, execute_path, execute_args,
                    experiment_results_dir, task_no, total_tasks):
    execute_string = f'bash {execute_path} {execute_args} ' +\
                    f'{experiment_results_dir}'
    console.log(f'Executing command: {execute_string}\n')
    status.update(
            f"[bold]Task [{task_no}/{total_tasks}]:[/bold] Experiment "
            f"{args.experiment_folder} with parameters {execute_args}.")

    f_stdout = open(os.path.join(experiment_results_dir, 'output.txt'), 'a')
    f_stderr = open(os.path.join(experiment_results_dir, 'error.txt'), 'a')
    f_stdout.write(f'Doing task {task_no}/{total_tasks}: {execute_string}\n')
    f_stderr.write(f'Doing task {task_no}/{total_tasks}: {execute_string}\n')

    try:
        run([execute_path, execute_args, experiment_results_dir],
            stdout=f_stdout, stderr=f_stderr)
    except Exception as e:
        console.log('[bold red]Something went wrong. Check log files!')
        console.log(e)

    f_stdout.write('\n\n')
    f_stderr.write('\n\n')
    f_stdout.close()
    f_stderr.close()
    console.log(f'Task done [{task_no}/{total_tasks}].\n')


def execute_job(console, status, execute_path, execute_args,
                experiment_results_dir, task_no, total_tasks,
                submit_info):
    info = copy.deepcopy(submit_info)
    info['output'] = os.path.join(experiment_results_dir, f'output_{task_no}.txt')
    info['error'] = os.path.join(experiment_results_dir, f'error_{task_no}.txt')
    info['log'] = os.path.join(experiment_results_dir, f'log_{task_no}.txt')
    info['arguments'] = f"{execute_args} {experiment_results_dir}"

    job = htcondor.Submit(info)
    console.log(f'Info on job [bold]{task_no}[/bold]:')
    console.log(str(job).strip())
    schedd = htcondor.Schedd()  # get Python representation of the scheduler
    submit_result = schedd.submit(job)  # submit the job

    console.log(
            f'Job [{task_no}/{total_tasks}] submitted to '
            f'Cluster {submit_result.cluster()}\n')
    return submit_result.cluster()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                description="Helper script to manage experiments "
                "for the project of online change detection with "
                "structure"
            )
    parser.add_argument('experiment_folder', help="Experiment folder path. "
                        "Must contains at least one script named execute.sh "
                        "performing the experiment.\n"
                        "It can take command line arguments "
                        "as an input to handle varying hyperparameters. "
                        "Those are specified thanks to the optional "
                        "argument --execute_args.\n"
                        "The results directory for this experiment will also "
                        "be passed as the last argument to handle data saving."
                        " Additionally, any number of scripts can be added to "
                        "perform actions after the experiment ended using the "
                        "syntax action_*.sh. in the name of the script."
                        )
    parser.add_argument('--runner', choices=['local', 'job'], default='local',
                        help="Specifies how to execute the experiment.\n "
                        "If job is specified. It is possible to specify its "
                        "options using args --n_cpus and --memory."
                        )
    parser.add_argument('--results_dir', type=str, default='results/',
                        help="Directory where experiment results are stored."
                        " Default to ./results/."
                        )
    parser.add_argument('--execute_args', default=[], action="append",
                        help="Arguments to pass to "
                        "execute.sh script. MUST BE USED WITH \"\" around.\n"
                        "For example:\n python launch_experiment.py "
                        "experiments/test --execute_args \"--arg1 arg1\".\n"
                        "It is also possible to specify multiple execute_args, "
                        "which will be executed as part of the same experiment. "
                        "If execution is local, they will be run one after the other, "
                        "Otherwhise the jobs will be launched simulataneously.",
                        )
    parser.add_argument('--n_cpus', type=int, default='1',
                        help="Number of cpus to ask for a job."
                        )
    parser.add_argument('--memory', type=str, default='100MB',
                        help="Memory to ask for a job."
                        )
    parser.add_argument('--is_flash', action='store_true', default=False,
                        help='Whether the job is short (Less than hour to have priority scheduling.')
    parser.add_argument('--ignore_git', action='store_true', default=False,
                        help='Wheter to ignore git commit requirements. NOT RECOMMENDED.')
    parser.add_argument('--tag', default=[], action='append',
                        help='Tag to reference the experiment. Can be used multiple times.')
    args = parser.parse_args()

    # Handling terminal info output using rich console
    console = Console()
    with console.status(
            f"[bold]Experiment {args.experiment_folder}: Setting things up"
            ) as status:
        try:
            repo = git.Repo('.')
            try:
                assert not repo.is_dirty()
            except AssertionError:
                console.log('[bold red]Repertory is not committed.')
                console.log(
                        'Please commit changes before launching an experiment')
                if args.ignore_git:
                    console.log(
                            'Git requirements overidden. '
                            'Taking previous commit as reference.')
                else:
                    sys.exit(0)

            # Handling experiment directory
            execute_path = os.path.join(args.experiment_folder, 'execute.sh')
            if not os.path.exists(execute_path):
                console.log(
                        f'[bold red]Execute script {execute_path} not found!')
                console.log('Quitting...')
                sys.exit(0)

            # Experiment database handling
            if not os.path.exists(args.results_dir):
                prompt_create_dir(args.results_dir, console, status)

            dB_path = os.path.join(args.results_dir, 'experiments.json')
            console.log(f'Database of experiments stored in: {dB_path}')
            dB = TinyDB(dB_path)

            # Handling experiment results directory
            if len(dB) == 0:
                experiment_id = 1
            else:
                experiment_id = dB.all()[-1]['id'] + 1
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

            console.log('Adding experiment '
                        f'[bold]{experiment_id}[/bold] to database')
            dB.insert({
                    'id': experiment_id,
                    'experiment_folder': args.experiment_folder,
                    'status': 'not started',
                    'arguments': args.execute_args,
                    'launch_date': datetime.datetime.now().isoformat(),
                    'experiment_results_dir': experiment_results_dir,
                    'commit_sha': repo.head.object.hexsha,
                    'tags': args.tag
                    })

            # Launching experiment
            if len(args.execute_args) == 0:
                args.execute_args = ['']
            dB.update({"status": "running"}, Query().id == experiment_id)

            if args.runner == "local":
                console.log(f'Launching {len(args.execute_args)} executions '
                            'from local runner\n')
                for task_no, execute_args in enumerate(args.execute_args):
                    execute_locally(console, status, execute_path,
                                    execute_args, experiment_results_dir,
                                    task_no+1, len(args.execute_args))
                dB.update({"status": "finished"}, Query().id == experiment_id)

            elif htcondor_available:
                console.log('Submitting job(s) to HTCondor')
                console.log(
                        f'Launching {len(args.execute_args)} executions '
                        'from HTCondor jobs\n')

                cluster_ids = []
                submit_info = {
                    'executable': execute_path,
                    'request_cpus': args.n_cpus,
                    'request_memory': args.memory,
                    'getenv': True,
                    'should_transfer_files': 'IF_NEEDED',
                    'when_to_transfer_output': 'ON_EXIT',
                    '+WishedAcctGroup': "group_usmb.listic",
                    '+isFlash': args.is_flash,
                    'batch_name': str(
                        pathlib.PurePath(args.experiment_folder).name
                        )
                    }

                for task_no, execute_args in enumerate(args.execute_args):
                    cluster = execute_job(console, status, execute_path,
                                          execute_args,
                                          experiment_results_dir, task_no+1,
                                          len(args.execute_args), submit_info)
                    cluster_ids.append(cluster)

                submit_info['cluster_ids'] = cluster_ids
                dB.update({
                    "submit_info": submit_info
                    }, Query().id == experiment_id)

            else:
                console.log('[bold red]Sorry impossible to run as job. '
                            'HTCondor is unaivalaible.')
        except KeyboardInterrupt:
            sys.exit(0)
