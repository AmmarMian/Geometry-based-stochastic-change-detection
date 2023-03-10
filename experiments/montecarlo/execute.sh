#!/usr/bin/bash
# ========================================
# FileName: execute.sh
# Date: 10 mars 2023 - 12:07
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: <brief>
# =========================================

echo "This is a testing script $0"
echo "Now I will sleep $1 seconds"
sleep $1
echo "Done. Quitting..."

python3 -c "import plotext as plt; plt.scatter(plt.sin()); plt.title('Scatter Plot'); plt.show();"
