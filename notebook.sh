#!/bin/bash
export PROJECT_ROOT=`pwd`
export PYTHONPATH=$PYTHONPATH:`pwd`/src
jupyter notebook $@