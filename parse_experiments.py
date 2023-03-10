# ========================================
# FileName: parse_experiments.py
# Date: 10 mars 2023 - 15:29
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Utility script to parse experiments
# launched from the launch_experiment.py script
# =========================================

import sys
import os
from tinydb import TinyDB, Query
from rich import print as rprint
from rich.console import Console
from rich.table import Table
import argparse
from simple_term_menu import TerminalMenu
import mmap
import pydoc


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
    return index + 1


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


def show_all_experiments(dB):
    console_table = Console()

    table = Table(title='Experiments')
    table.add_column('ID', justify='left', style='white', no_wrap=True)
    table.add_column('Tags', justify='center', style='Yellow', no_wrap=True)
    table.add_column('Experiment folder', justify='center', style='blue', no_wrap=True)
    table.add_column('Start date', justify='center', style='magenta', no_wrap=True)
    table.add_column('Status', justify='right', no_wrap=True)

    for experiment in dB:
        table.add_row(str(experiment['id']), ", ".join(experiment['tags']), 
                      experiment['experiment_folder'],
                      experiment['launch_date'], experiment['status'])
    console_table.print(table)

def menu_experiment(experiment):
    rprint(f"[bold]Showing information for experiment {experiment['id']}\n")

    # Action menu on this specific experiment
    rprint(experiment)

def select_experiment(dB):
    choices = ['[a] Select from ID', '[b] Select from menu']
    menu = TerminalMenu(choices)
    choice = menu.show()

    if choice == 0:
        ID = input('Enter an ID: ')
    elif choice == 1:
        ID = select_experiment_menu(dB)

    experiment = dB.search(Query().id == int(ID))
    if len(experiment) == 1:
        menu_experiment(experiment[0])
    else:
        print('Sorry experiment not found, try again..')

def main_menu(dB):

    choice = None
    choices = [
            '[a] Show all experiments',
            '[b] Select an experiment',
            '[c] Filter experiments',
            '[q] quit'
            ]
    menu = TerminalMenu(choices)
    while choice != len(choices) - 1:
        choice = menu.show()

        if choice == 0:
            show_all_experiments(dB)
        elif choice == 1:
            select_experiment(dB)
        elif choice == 2:
            filter_experiments(dB)
    sys.exit(0)


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
    main_menu(dB)
