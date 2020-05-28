@echo off

setlocal
set PROJECT_ROOT=%cd%
set PYTHONPATH=%PYTHONPATH%;%cd%/src

call conda activate genomics
call jupyter notebook
