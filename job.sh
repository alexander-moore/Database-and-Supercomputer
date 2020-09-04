#!/bin/sh
#sbatch -n4 -N1 --pty

import torch

python python_script.py
