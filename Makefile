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
Realdata: # Simulations on real UAVSAR data (do targets Scene1-crop, Scene2-crop, Scene3-crop, Scene4-cropmediumtemporal, Scene4-crophightemporal)
	@make Scene1-crop
	@make Scene2-crop
	@make Scene3-crop
	@make Scene4-cropmediumtemporal
	@make Scene4-crophightemporal


.PHONY: Scene1-full
Scene1-full: # Simulations on real UAVSAR data, Scene1. Repeated 5 times temporally.
	@{ \
	CMD="python launch_experiment.py experiments/real_data"; \
	for DETECTOR in "gaussian_glrt" "scaled_gaussian_glrt" "scaled_gaussian_sgd" "scaled_gaussian_kron_glrt" "scaled_gaussian_kron_sgd" ; do\
		CMD="$$CMD --execute_args \"1 5 -d $$DETECTOR --data_path $$DATA_PATH\"" ;\
	done; \
	CMD="$$CMD --runner $$RUNNER --n_cpus 4 --memory 8GB --tag scene1 --tag repeat --tag nocrop";\
	echo "Evaluating command: $$CMD"; \
	eval $$CMD; \
	}


.PHONY: Scene1-crop
Scene1-crop: # Simulations on real UAVSAR data, Scene1. Cropped and repeated 10 times temporally.
	@{ \
	CMD="python launch_experiment.py experiments/real_data"; \
	for DETECTOR in "gaussian_glrt" "scaled_gaussian_glrt" "scaled_gaussian_sgd" "scaled_gaussian_kron_glrt" "scaled_gaussian_kron_sgd" ; do\
		CMD="$$CMD --execute_args \"1 10 -c 0 500 0 500 -d $$DETECTOR --data_path $$DATA_PATH\"" ;\
	done; \
	CMD="$$CMD --runner $$RUNNER --n_cpus 4 --memory 8GB --tag scene1 --tag repeat --tag crop";\
	echo "Evaluating command: $$CMD"; \
	eval $$CMD; \
	}


.PHONY: Scene2-full
Scene2-full: # Simulations on real UAVSAR data, Scene2. Repeated 5 times temporally.
	@{ \
	CMD="python launch_experiment.py experiments/real_data"; \
	for DETECTOR in "gaussian_glrt" "scaled_gaussian_glrt" "scaled_gaussian_sgd" "scaled_gaussian_kron_glrt" "scaled_gaussian_kron_sgd" ; do\
		CMD="$$CMD --execute_args \"2 5 -d $$DETECTOR--data_path $$DATA_PATH\"" ;\
	done; \
	CMD="$$CMD --runner $$RUNNER --n_cpus 4 --memory 8GB --tag scene2 --tag repeat --tag nocrop";\
	echo "Evaluating command: $$CMD"; \
	eval $$CMD; \
	}


.PHONY: Scene2-crop
Scene2-crop: # Simulations on real UAVSAR data, Scene1. Cropped and repeated 10 times temporally.
	@{ \
	CMD="python launch_experiment.py experiments/real_data"; \
	for DETECTOR in "gaussian_glrt" "scaled_gaussian_glrt" "scaled_gaussian_sgd" "scaled_gaussian_kron_glrt" "scaled_gaussian_kron_sgd" ; do\
		CMD="$$CMD --execute_args \"2 10 -c 0 500 0 500 -d $$DETECTOR --data_path $$DATA_PATH\"" ;\
	done; \
	CMD="$$CMD --runner $$RUNNER --n_cpus 4 --memory 8GB --tag scene2 --tag repeat --tag crop";\
	echo "Evaluating command: $$CMD"; \
	eval $$CMD; \
	}


.PHONY: Scene3-crop
Scene3-crop: # Simulations on real UAVSAR data, Scene3. Cropped and repeated 5 times temporally.
	@{ \
	CMD="python launch_experiment.py experiments/real_data"; \
	for DETECTOR in "gaussian_glrt" "scaled_gaussian_glrt" "scaled_gaussian_sgd" "scaled_gaussian_kron_glrt" "scaled_gaussian_kron_sgd" ; do\
		CMD="$$CMD --execute_args \"3 5 -c 0 500 0 750 -d $$DETECTOR --data_path $$DATA_PATH\"" ;\
	done; \
	CMD="$$CMD --runner $$RUNNER --n_cpus 4 --memory 16GB --tag scene3 --tag repeat --tag crop";\
	echo "Evaluating command: $$CMD"; \
	eval $$CMD; \
	}


.PHONY: Scene4-crophightemporal
Scene4-crophightemporal: # Simulations on real UAVSAR data, Scene4. Cropped on the smallest patch and repeated 10 times temporally.
	@{ \
	CMD="python launch_experiment.py experiments/real_data"; \
	for DETECTOR in "gaussian_glrt" "scaled_gaussian_glrt" "scaled_gaussian_sgd" "scaled_gaussian_kron_glrt" "scaled_gaussian_kron_sgd" ; do\
		CMD="$$CMD --execute_args \"4 10 -c 2800 3000 1600 1800 -d $$DETECTOR --data_path $$DATA_PATH\"" ;\
	done; \
	CMD="$$CMD --runner $$RUNNER --n_cpus 4 --memory 32GB --tag scene4 --tag repeat --tag crop --tag hightemporal";\
	echo "Evaluating command: $$CMD"; \
	eval $$CMD; \
	}


.PHONY: Scene4-cropmediumtemporal
Scene4-cropmediumtemporal: # Simulations on real UAVSAR data, Scene4. Cropped on a small patch and repeated 5 times temporally.
	@{ \
	CMD="python launch_experiment.py experiments/real_data"; \
	for DETECTOR in "gaussian_glrt" "scaled_gaussian_glrt" "scaled_gaussian_sgd" "scaled_gaussian_kron_glrt" "scaled_gaussian_kron_sgd" ; do\
		CMD="$$CMD --execute_args \"4 5 -c 2500 3000 1400 2200 -d $$DETECTOR --data_path $$DATA_PATH\"" ;\
	done; \
	CMD="$$CMD --runner $$RUNNER --n_cpus 4 --memory 32GB --tag scene4 --tag repeat --tag crop --tag mediumtemporal";\
	echo "Evaluating command: $$CMD"; \
	eval $$CMD; \
	}


.PHONY: help
help: # Show the help message
	@echo "This is a Makefile for project: ${PROJECT_NAME}"
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done
