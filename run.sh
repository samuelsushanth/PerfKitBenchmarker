#!/bin/bash

cloud=$1
~/git/PerfKitBenchmarker/pkb.py --benchmarks=cuda_hpl --benchmark_config_file=/usr/local/google/home/ferneyhough/git/PerfKitBenchmarker/cuda_hpl_config.yml --flag_matrix=$cloud --cloud=$cloud
