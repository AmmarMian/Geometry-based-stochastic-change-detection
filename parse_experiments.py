# ========================================
# FileName: parse_experiments.py
# Date: 10 mars 2023 - 15:29
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Utility script to parse experiments
# launched from the launch_experiment.py script
# =========================================

import os
from tinydb import TinyDB, Query
import rich
import argparse
from simple_term_menu import TerminalMenu
import mmap

def select_experiment_menu(dB):
    choices = [
            f"{experiment['id']}: {experiment['experiment_folder']} - start: {experiment['launch_date']} - status: {experiment['status']}" 
            for experiment in dB
            ]

    def preview_command(entry):
        id = int(entry.split(":")[0])
        experiment = dB.all()[id-1]
        with open(os.path.join(experiment['experiment_results_dir'], 'output.txt'), 'r') as f:
            file_content = f.read()
        return file_content

    menu = TerminalMenu(choices, title='Select an experiment:',
                        preview_command=preview_command,
                        preview_title='Experiment output',
                        preview_size=2)
    index = menu.show()
    return index


def update_status_jobs(dB):
    """Update status of HTCondor jobs that have terminated"""
    experiment_as_jobs = dB.search(Query().submit_info.exists())
    for experiment in experiment_as_jobs:

        # Search efficiently in log file:
        # https://pynative.com/python-search-for-a-string-in-text-files/
        # Accessed 10/03/23 16:47
        with open(os.path.join(experiment["experiment_results_dir"], "log.txt"), 'rb', 0) as file:
            s = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            if s.find(b'Job terminated') != -1:
                dB.update({
                    'status': 'finished'
                    }, Query().id == experiment['id'])
    return dB

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='Helper script to visualize and compare experiments')
    parser.add_argument('--results_dir', default='./results',
                        help='Directory where experiment database and results are stored')
    args = parser.parse_args()

    # Reading database into memory
    dB = TinyDB(os.path.join(args.results_dir, 'experiments.json'))
    dB = update_status_jobs(dB)

    # Main menu
    print(select_experiment_menu(dB))
