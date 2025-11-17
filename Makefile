#########################################################################################
# GLobal Variables																		#
#########################################################################################

PROJECT_NAME = dl_hspc
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#########################################################################################
# COMMANDS																				#
#########################################################################################

## install Python dependencies
.PHONY: requirements
requirements:
	mamba env update --name $(PROJECT_NAME) --file environment.yml --prune



## delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8 and black (use 'make format' to apply formatting)
.PHONY: lint
lint:
	flake8 dl_hspc
	black dl_hspc --check --config pyproject.toml hspc

## Format source code using black
.PHONY: format
format:
	black dl_hspc --config pyproject.toml dl_hspc


## setup python interpreter environment
.PHONY: create_environment
create_environment:
	mamba env create --name $(PROJECT_NAME) --file environment.yml

	@echo ">>> conda env created."


#########################################################################################
# PROJECT RULES																			#
#########################################################################################

## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) dl_hspc/data/make_dataset.py


#########################################################################################
# Self Documenting Commands																#
#########################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n' .join([line for line in sys.stdin]); \
matches = re.findall(r''\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines) \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)