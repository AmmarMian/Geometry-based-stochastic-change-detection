# ========================================
# FileName: submit_jobs_montecarlo.py
# Date: 28 février 2023 - 15:15
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Submit job for monte-carlo 
#        simulations.
# =========================================

import os
# import htcondor
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Monte-carlo job submission utility")
    parser.add_argument("scneario", help="Type of CD scenario",
                        choices=['change', 'nochange'])
    parser.add_argument("p", help="dimension of data",
                        type=int)
    parser.add_argument("N", help="number of samples at each batch",
                        type=int)
    parser.add_argument("n_batches", help="Number of batches of data",
                        type=int)
    parser.add_argument("nu_before", help="scale factor before change",
                        type=float)
    parser.add_argument("nu_after", help="scale factor after change, "
                        "only when scenario is change",
                        type=float)
    parser.add_argument("rho_before", help="correlation factor before change",
                        type=float)
    parser.add_argument("nu_after", help="correlation factor after change,"
                        "only when scenario is change",
                        type=float)
    parser.add_argument("n_points", help="Number of points to plot",
                        type=int, default=10)
    args = parser.parse_args()
