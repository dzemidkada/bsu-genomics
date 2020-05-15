#!/bin/bash
export PROJECT_ROOT=`pwd`
export PYTHONPATH=$PYTHONPATH:`pwd`/src
python src/dash_app/map/app.py $@