# ========================================
# FileName: Makefile
# Date: 14 mars 2023 - 08:47
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: <brief>
# =========================================
include CONFIG.sh

SHELL := /bin/bash
.DEFAULT_GOAL:=help


.PHONY: Realdata
Realdata: # Simulations on real UAVSAR data
	@make Scene1
	@make Scene2


.PHONY: Scene1
Scene1: # Simulations on real UAVSAR data, Scene1
	@{ \
	CMD="python launch_experiment.py experiment/realdata"; \
	for DETECTOR in "gaussian_glrt" "scaled_gaussian_glrt" "scaled_gaussian_sgd" "scaled_gaussian_kron_glrt" "scaled_gaussian_kron_sgd" ; do\
		CMD="$$CMD --execute_args \"1 10 -d $$DETECTOR\"" ;\
	done; \
	CMD="$$CMD --runner job --n_cpus 8 --memory 2GB --tag scene1 --tag repeat --tag nocrop";\
	echo "Evaluating command: $$CMD"; \
	eval $$CMD; \
	}


.PHONY: Scene2
Scene2: # Simulations on real UAVSAR data, Scene1
	@echo 'Test 2'

.PHONY: help
help: # Show the help message
	@echo "This is a Makefile for project: ${PROJECT_NAME}"
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done
