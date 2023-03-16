#!/usr/bin/bash
# ========================================
# FileName: execute.sh
# Date: 10 mars 2023 - 12:07
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: <brief>
# =========================================

python - << EOF
import pickle
import numpy as np
import os
import plotext

results_dir = "${BASH_ARGV[0]}"
data = np.random.randn(100)
with open(os.path.join(results_dir, f'artifact_{${BASH_ARGV[1]}}.pkl'), 'wb') as f:
    pickle.dump(data, f)

plotext.plot(data)
plotext.show()


EOF
