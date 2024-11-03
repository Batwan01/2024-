#!/bin/bash

pip install -U openmim 
mim install mmengine 
mim install "mmcv==2.1.0" 
mim install mmdet


pip install importlib-metadata
pip install platformdirs tomli
pip install "numpy<2"
pip install fairscale
