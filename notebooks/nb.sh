#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT=$(readlink -f "${SCRIPT_DIR}/..")

export PYTHONPATH="${PYTHONPATH}:${ROOT}"
jupyter notebook
