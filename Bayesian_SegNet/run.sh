#!/bin/sh
mkdir -p models
python3 -m jupyter nbconvert --output-dir ./models --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=sys_python36 --to notebook --execute $1
