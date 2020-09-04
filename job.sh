#!/bin/sh
#sbatch -n4 -N1 --pty /bin/bash
python python_script.py
