# Copyright 2016 PerfKitBenchmarker Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs NVIDIA's CUDA Accelerated HPL.

Download page (registration required):
https://developer.nvidia.com/accelerated-computing-developer-program-home
Note that the tarball must be downloaded manually and placed in PKB's data
folder. See instructions in linux_packagtes/cuda_hpl.py for more information.

HPL Homepage: http://www.netlib.org/benchmark/hpl/

HPL requires a BLAS library (Basic Linear Algebra Subprograms)
OpenBlas: http://www.openblas.net/

HPL also requires a MPI (Message Passing Interface) Library
OpenMPI: http://www.open-mpi.org/

MPI needs to be configured:
Configuring MPI:
http://techtinkering.com/2009/12/02/setting-up-a-beowulf-cluster-using-open-mpi-on-linux/

Once HPL is built the configuration file must be created:
Configuring HPL.dat:
http://www.advancedclustering.com/faq/how-do-i-tune-my-hpldat-file.html
http://www.netlib.org/benchmark/hpl/faqs.html
"""

import logging
import math
import re
import os
import ipdb

from perfkitbenchmarker import configs
from perfkitbenchmarker import data
from perfkitbenchmarker import flags
from perfkitbenchmarker import regex_util
from perfkitbenchmarker import sample
from perfkitbenchmarker import vm_util
from perfkitbenchmarker.linux_packages import cuda_hpl
from perfkitbenchmarker.linux_packages import cuda_toolkit_8

FLAGS = flags.FLAGS
LOCAL_HPL_CONFIG_FILE = 'cuda_hpl_config.txt'
REMOTE_HPL_CONFIG_FILE = 'HPL.dat'
MACHINEFILE = 'machinefile'
BLOCK_SIZE = 1024

BENCHMARK_NAME = 'cuda_hpl'
BENCHMARK_CONFIG = """
cuda_hpl:
  description: Runs CUDA HPL. Specify the number of VMs with --num_vms
  flags:
    gce_migrate_on_maintenance: False
  vm_groups:
    default:
      vm_spec:
        GCP:
          image: ubuntu1604-cuda-hpl
          machine_type: n1-standard-8-k80x2
          zone: us-east1-d
          boot_disk_size: 200
        AWS:
          image: ami-a9d276c9
          machine_type: p2.xlarge
          zone: us-west-2b
          boot_disk_size: 200
        Azure:
          image: Canonical:UbuntuServer:16.04.0-LTS:latest
          machine_type: Standard_NC6
          zone: eastus
"""

flags.DEFINE_integer('cuda_hpl_memory_size_mb',
                     None,
                     'The amount of memory in MB on each machine to use. By '
                     'default it will use the entire system\'s memory.')


def GetConfig(user_config):
  return configs.LoadConfig(BENCHMARK_CONFIG, user_config, BENCHMARK_NAME)


def CheckPrerequisites(benchmark_config):
  """Verifies that the required resources are present.

  Raises:
    perfkitbenchmarker.data.ResourceNotFound: On missing resource.
  """
  data.ResourcePath(LOCAL_HPL_CONFIG_FILE)


def CreateMachineFile(vms):
  """Create a file with the IP of each machine in the cluster on its own line.

  Args:
    vms: The list of vms which will be in the cluster.
  """
  with vm_util.NamedTemporaryFile() as machine_file:
    master_vm = vms[0]
    machine_file.write('localhost slots=%d\n' % master_vm.num_cpus)
    for vm in vms[1:]:
      machine_file.write('%s slots=%d\n' % (vm.internal_ip,
                                            vm.num_cpus))
    machine_file.close()
    master_vm.PushFile(machine_file.name, MACHINEFILE)


def CalculateGpuToCpuFlopsRatio():
  tesla_k80_gpu_flops = 1.455 * 1e12


def GenerateHplConfiguration(vm, benchmark_spec):
  """Create the HPL configuration file."""
  num_vms = len(benchmark_spec.vms)
  assert num_vms == 1
  if FLAGS.cuda_hpl_memory_size_mb:
    total_memory = FLAGS.cuda_hpl_memory_size_mb * 1024 * 1024 * num_vms
  else:
    # Sum of Free, Cached, Buffers in kb
    stdout, _ = vm.RemoteCommand("""
      awk '
        BEGIN      {total =0}
        /MemFree:/ {total += $2}
        /Cached:/  {total += $2}
        /Buffers:/ {total += $2}
        END        {print total}
        ' /proc/meminfo
        """)
    available_memory = int(stdout)
    total_memory = available_memory * 1024 * num_vms
  total_gpus = cuda_toolkit_8.QueryNumberOfGpus(vm) * num_vms
  block_size = BLOCK_SIZE

  # Finds a problem size that will fit in memory and is a multiple of the
  # block size.
  base_problem_size = math.sqrt(total_memory * .1)
  blocks = int(base_problem_size / block_size)
  blocks = blocks if (blocks % 2) == 0 else blocks - 1
  problem_size = block_size * blocks

  # Makes the grid as 'square' as possible, with rows < columns
  sqrt_gpus = int(math.sqrt(total_gpus)) + 1
  num_rows = 0
  num_columns = 0
  for i in reversed(range(sqrt_gpus)):
    if total_gpus % i == 0:
      num_rows = i
      num_columns = total_gpus / i
      break

  local_file_path = data.ResourcePath(LOCAL_HPL_CONFIG_FILE)
  remote_file_path = REMOTE_HPL_CONFIG_FILE
  vm.PushFile(local_file_path, remote_file_path)
  sed_cmd = (('sed -i -e "s/problem_size/%s/" -e "s/block_size/%s/" '
              '-e "s/rows/%s/" -e "s/columns/%s/" %s') %
             (problem_size, block_size, num_rows, num_columns, remote_file_path))
  vm.RemoteCommand(sed_cmd)


def PrepareCudaHpl(vm):
  """Builds CUDA HPL on a single vm."""
  logging.info('Building CUDA HPL on %s', vm)
  vm.Install('cuda_hpl')


def Prepare(benchmark_spec):
  """Install CUDA HPL on the target vms.

  Args:
    benchmark_spec: The benchmark specification. Contains all data that is
        required to run the benchmark.
  """
  vms = benchmark_spec.vms
  master_vm = vms[0]

  PrepareCudaHpl(master_vm)
  GenerateHplConfiguration(master_vm, benchmark_spec)
  #CreateMachineFile(vms)


def UpdateMetadata(metadata):
  """Update metadata with hpcc-related flag values."""
  metadata['memory_size_mb'] = FLAGS.memory_size_mb
  if FLAGS['hpcc_binary'].present:
    metadata['override_binary'] = FLAGS.hpcc_binary
  if FLAGS['hpcc_mpi_env'].present:
    metadata['mpi_env'] = FLAGS.hpcc_mpi_env


def ParseOutput(hpl_output, benchmark_spec, cpus_used):
  """Parses the output from HPL.

  Args:
    hpcc_output: A string containing the hpl output. 
    benchmark_spec: The benchmark specification. Contains all data that is
        required to run the benchmark.
    cpus_used: Number of physical CPU cores used to run the benchmark. 

  Returns:
    A list of samples to be published (in the same format as Run() returns).
  """
  hpl_output_lines = hpl_output.splitlines()
  find_results_header_regex = r'T\/V\s+N\s+NB\s+P\s+Q\s+Time\s+Gflops\s*$'
  for idx, line in enumerate(hpl_output_lines):
    if re.match(find_results_header_regex, line):
      results_line_idx = idx + 2
      break

  hpl_results = hpl_output_lines[results_line_idx].split()
  metadata = dict()
  metadata['num_machines'] = len(benchmark_spec.vms)
  metadata['N'] = int(hpl_results[1])
  metadata['NB'] = int(hpl_results[2])
  metadata['P'] = int(hpl_results[3])
  metadata['Q'] = int(hpl_results[4])
  metadata['cpus_used'] = cpus_used
  metadata['test_version'] = "0.1"
  UpdateMetadata(metadata)
  
  flops = float(hpl_results[6])
  results = [sample.Sample('HPL Throughput', flops, 'Gflops', metadata)]
  return results


#def GenerateUpdateRunLinpackSedCmd(num_cpus, num_gpus):
#  cpus_per_gpu = num_cpus / num_gpus
#  run_linpack_path = os.path.join(cuda_hpl.HPL_BIN_DIR,
#                                  'run_linpack')
#  # gpu_flops / cpu_flops per core * num_core_per_gpu + gpu_flops
#  cuda_dgemm_split = 2910 / (2.3 * 16 * cpus_per_gpu + 2910)
#  cuda_dtrsm_split = cuda_dgemm_split - 0.1
#  sed_cmd = (('sed -i -e "s/\(CPU_CORES_PER_GPU=\).*/\1%s/" '
#              '-e "s/\(CUDA_DGEMM_SPLIT=\).*/\1%s/" '
#              '-e "s/\(CUDA_DTRSM_SPLIT=\).*/\1%s/" %s') %
#             (cpus_per_gpu, cuda_dgemm_split, cuda_dtrsm_split,
#              run_linpack_path))
#  return sed_cmd
#
#
#def UpdateRunLinkpackConfig(num_cpus, num_gpus, vm):
#  vm.RemoteCommand(GenerateUpdateRunLinpackSedCmd(num_cpus, num_gpus))


def Run(benchmark_spec):
  """Run HPCC on the cluster.

  Args:
    benchmark_spec: The benchmark specification. Contains all data that is
        required to run the benchmark.

  Returns:
    A list of sample.Sample objects.
  """
  vms = benchmark_spec.vms
  master_vm = vms[0]
  run_linpack_path = os.path.join(cuda_hpl.HPL_BIN_DIR,
                                   'run_linpack')
  num_gpus = cuda_toolkit_8.QueryNumberOfGpus(master_vm) * len(vms)
  #num_cpus = master_vm.num_cpus
  mpi_cmd = ('mpirun -np %s %s' %
             (num_gpus, run_linpack_path))
  
  results = []
  #num_cpus_to_use = num_cpus
  #UpdateRunLinkpackConfig(num_cpus_to_use, num_gpus, master_vm)
  master_vm.RemoteCommand(mpi_cmd)
  logging.info('CUDA HPL Results:')
  run_results, _ = master_vm.RemoteCommand('cat HPL.out', should_log=True)
  results.extend(ParseOutput(run_results, benchmark_spec, num_cpus_to_use))

  return results


def Cleanup(benchmark_spec):
  """Cleanup HPCC on the cluster.

  Args:
    benchmark_spec: The benchmark specification. Contains all data that is
        required to run the benchmark.
  """
  vms = benchmark_spec.vms
  master_vm = vms[0]
  master_vm.RemoveFile('hpcc*')
  master_vm.RemoveFile(MACHINEFILE)

  for vm in vms[1:]:
    vm.RemoveFile('hpcc')
    vm.RemoveFile('/usr/bin/orted')
