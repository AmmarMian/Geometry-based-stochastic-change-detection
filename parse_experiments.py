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
import re
from subprocess import run
import shutil


def select_experiment_menu(dB):
    """Menu to select an experiment among all available.
    With a preview of the output.

    Args:
        dB (TinyDB): Database of experiments

    Returns:
        int: index in the database of the choice
    """
    choices = [
            f"{experiment['id']}: {experiment['experiment_folder']} - "
            f"start: {experiment['launch_date']} - "
            f"status: {experiment['status']}"
            for experiment in dB
        ]

    def preview_command(entry):
        ID = int(entry.split(":")[0])
        experiment = dB.search(Query().id == ID)[0]

        # Find all output files
        # rx = re.compile("output(.*)txt")
        # search = list(os.walk(experiment['experiment_results_dir']))[0]
        # list_files = [file for file in search[-1]
                      # if rx.match(file)]
        # file_content = ""
        # for file in list_files:
            # file_content += f"File {file}:\n"
            # with open(os.path.join(
                # experiment['experiment_results_dir'], file), 'r') as f:
                # file_content += f.read()
            # file_content += "\n"
        # return file_content
        return concatenate_files_match(
                experiment['experiment_results_dir'], 'output(.*)txt'
                )

    # def preview_command(entry):
        # ID = int(entry.split(":")[0])
        # experiment = dB.search(Query().id == ID)[0]
        # results_path = experiment['experiment_results_dir']
        # return "\n".join(os.listdir(results_path))

    menu = TerminalMenu(choices, title='Select an experiment: (press q for quitting)',
                        preview_command=preview_command,
                        preview_title='Experiment output',
                        preview_size=2)
    index = menu.show()

    if index is None:
        return None
    else:
        return index + 1


def update_status_jobs(dB):
    """Update status of HTCondor jobs that have terminated"""
    experiment_as_jobs = dB.search(Query().submit_info.exists())
    for experiment in experiment_as_jobs:

        # Looking for log files
        rx = re.compile("log(.*)txt")
        search = list(os.walk(experiment['experiment_results_dir']))[0]

        list_logfiles = [file for file in search[-1]
                         if rx.match(file)]

        # Search efficiently in log file:
        # https://pynative.com/python-search-for-a-string-in-text-files/
        # Accessed 10/03/23 16:47
        job_is_over = True
        file_no = 0
        while job_is_over and file_no < len(list_logfiles):
            with open(os.path.join(experiment["experiment_results_dir"],
                      list_logfiles[file_no]), 'rb', 0) as file:
                s = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
                job_is_over = s.find(b'Job terminated') != -1
                file_no += 1
        if job_is_over:
            dB.update({
                'status': 'finished'
                }, Query().id == experiment['id'])
    return dB


def show_all_experiments(dB):
    """Action to show in a table all available experiments.

    Args:
        dB (TinydB): Experiments database
    """
    dB = update_status_jobs(dB)
    console_table = Console()

    table = Table(title='Experiments')
    table.add_column('ID', justify='left', style='white', no_wrap=True)
    table.add_column('Tags', justify='center', style='Yellow', no_wrap=True)
    table.add_column('Experiment folder', justify='center', style='blue', no_wrap=True)
    table.add_column('Arguments', justify='center', style='cyan', no_wrap=True)
    table.add_column('Start date', justify='center', style='magenta', no_wrap=True)
    table.add_column('Status', justify='right', no_wrap=True)

    for experiment in dB:
        table.add_row(str(experiment['id']), ", ".join(experiment['tags']), 
                      experiment['experiment_folder'], ",\n".join(experiment['arguments']),
                      experiment['launch_date'], experiment['status'])
    console_table.print(table)


def list_actions_available(experiment):
    """Parse from experiment directory the bash scripts corresponding
    to available actions.

    Args:
        experiment (TinydB): Experiments database
    """
    rx = re.compile("action_(.*)sh$")
    search = list(os.walk(experiment['experiment_folder']))[0]
    return [os.path.join(experiment['experiment_folder'], file)
            for file in search[-1]
            if rx.match(file)]


def parse_metadata_action(action_file_path):
    """Parsing header of action script file to show in the action menu.
    """
    with open(action_file_path, 'r') as f:
        f.readline()  # To skip the #!usr/bin/bash
        title = f.readline().split('Title: ')[-1].strip()
        description = f.readline().split('Description: ')[-1].strip()

    return title, description


def concatenate_files_match(path, regexp):
    """Concatenate all files correponding to a
    regexp in path directory and return a string."""

    rx = re.compile(regexp)
    search = list(os.walk(path))[0]
    list_files = [file for file in search[-1]
                  if rx.match(file)]
    file_content = ""
    for file in list_files:
        file_content += f"File {file}:\n"
        with open(os.path.join(path, file), 'r') as f:
            file_content += f.read()
        file_content += "\n"
    return file_content


def menu_experiment(experiment):

    # Printing experiment information
    rprint(f"[bold]Showing information for experiment {experiment['id']}")
    rprint(experiment)

    # Options to show logs, error and output
    error = concatenate_files_match(
            experiment['experiment_results_dir'], 'error(.*)txt'
            )
    output = concatenate_files_match(
            experiment['experiment_results_dir'], 'output(.*)txt'
            )
    log = concatenate_files_match(
            experiment['experiment_results_dir'], 'log(.*)txt'
            )
    choices = ['[a] See output(s)', '[b] See error(s)']

    if 'submit_info' in experiment:
        choices += ['[c] See logs']
        base_number = 100
    else:
        base_number = 99

    # Showing available actions
    metadata = {}
    actions = list_actions_available(experiment)
    for i, action in enumerate(actions):
        title, description = parse_metadata_action(action)
        metadata[title] = description
    choices += [f'[{chr(i+base_number)}] ' + title for title in metadata.keys()]

    def preview_command(entry):
        if entry == choices[0][4:]:
            return "See concatenation of all outputs from experiment"
        elif entry == choices[1][4:]:
            return "See concatenation of all errors files from experiment"
        elif entry == choices[2][4:] and 'submit_info' in experiment:
            return "See concatenation of all logs from experiment. (Only for a job runner.)"
        else:
            return metadata[entry]

    menu = TerminalMenu(choices, title="Select an action:",
                        preview_command= preview_command,
                        preview_title="Description",
                        preview_size=2)
    index = menu.show()
    while index is not None:
        if index == 0:
            print(output)
        elif index == 1:
            pydoc.pager(error)
        elif index == 2 and 'submit_info' in experiment:
            pydoc.pager(log)
        else:
            rprint(f"[bold green]Executing action {actions[index - base_number + 97]}")
            try:
                run([actions[index - base_number + 97],  # To take into account if submit or not
                     experiment['experiment_results_dir']])
            except KeyboardInterrupt:
                pass
        index = menu.show()


def select_experiment(dB):
    """Action relative to selecting a single experiment.

    Args:
        dB (TinydB): Experiments database
    """
    choices = ['[a] Select from ID', '[b] Select from menu']
    menu = TerminalMenu(choices)
    choice = menu.show()

    ID = None
    if choice == 0:
        ID = input('Enter an ID: ')
    elif choice == 1:
        ID = select_experiment_menu(dB)

    if ID is not None:
        experiment = dB.search(Query().id == int(ID))
        if len(experiment) == 1:
            menu_experiment(experiment[0])
        else:
            print('Sorry experiment not found, try again..')


def delete_experiment(dB, experiment_id):
    rprint(f"[bold red]You are going to delete experiment {experiment_id}")
    rprint(dB.search(Query().id == experiment_id)[0])

    choice = input('Are you sure (No, yes)? ')
    if choice == 'yes':

        # Removing experiment directory
        shutil.rmtree(
                dB.search(
                    Query().id == experiment_id
                    )[0]['experiment_results_dir']
                )

        # Removing experiment from database
        dB.remove(Query().id == experiment_id)
        rprint('[bold green]Successfully deleted experiment')


def delete_experiments_menu(dB):
    rprint('Experiment deletion menu')
    choices = ['[a] Select from ID', '[b] Select from menu']
    menu = TerminalMenu(choices)
    choice = menu.show()

    ID = None
    if choice == 0:
        ID = int(input('Enter an ID: '))
    elif choice == 1:
        ID = select_experiment_menu(dB)

    if choice is not None and len(dB.search(Query().id == ID)) == 1:
        delete_experiment(dB, ID)
    else:
        rprint('[bold red]ID has no or more than one match')


def filter_experiments(dB):
    print("Sorry, not implemented yet.")


def main_menu(dB):
    """Main menu at beginning of the script

    Args:
        dB (TinydB): Experiments database
    """
    choice = None
    choices = [
            '[a] Show all experiments',
            '[b] Select an experiment',
            '[c] Filter experiments',
            '[d] Delete an experiment',
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
        elif choice == 3:
            delete_experiments_menu(dB)
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
