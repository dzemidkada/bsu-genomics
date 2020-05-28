@echo off

setlocal
set PROJECT_ROOT=%cd%
set PYTHONPATH=%PYTHONPATH%;%cd%/src

call conda activate genomics
python src/dash_app/str_annotator/app.py $@
