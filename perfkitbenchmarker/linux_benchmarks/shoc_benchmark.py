# Copyright 2017 PerfKitBenchmarker Authors. All rights reserved.
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


"""Runs SHOC benchmark"""

import numpy
import re
import os
from perfkitbenchmarker import configs
from perfkitbenchmarker import flags
from perfkitbenchmarker import sample
from perfkitbenchmarker import regex_util
from perfkitbenchmarker import vm_util
from perfkitbenchmarker.linux_packages import shoc_benchmark_suite
from perfkitbenchmarker.linux_packages import cuda_toolkit_8
from perfkitbenchmarker import num_gpus_map_util


flags.DEFINE_integer('shoc_iterations', 5,
                     'number of iterations to run',
                     lower_bound=1)

flags.DEFINE_string('problem_size', '4096,4096',
                    'problem size')
#'19456,19456'
FLAGS = flags.FLAGS

NETWORK_DELAY_CMD = 'tc qdisc add dev ens5 root netem delay {}ms'
NETWORK_DELAY_TEARDOWN_CMD = 'tc qdisc del dev eth0 root'
MACHINEFILE = 'machinefile'
BENCHMARK_NAME = 'shoc'
BENCHMARK_VERSION = '0.24'
# Note on the config: gce_migrate_on_maintenance must be false,
# because GCE does not support migrating the user's GPU state.
BENCHMARK_CONFIG = """
shoc:
  description: Runs SHOC Benchmark Suite. Specify the number of VMs with --num_vms
  flags:
    gce_migrate_on_maintenance: False
  vm_groups:
    default:
      vm_spec:
        GCP:
          image: cuda-toolkit-8-shoc
          machine_type: n1-standard-4-k80x1
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
      vm_count: null
"""


def GetConfig(user_config):
  config = configs.LoadConfig(BENCHMARK_CONFIG, user_config, BENCHMARK_NAME)
  return config


def CheckPrerequisites(benchmark_config):
  """Verifies that the required resources are present.

  Raises:
    perfkitbenchmarker.data.ResourceNotFound: On missing resource.
  """
  pass


def AssertCorrectNumberOfGpus(vm):
  """Assert that VM is reporting the correct number of GPUs."""

  expected_num_gpus = num_gpus_map_util.gpus_per_vm[vm.machine_type]
  actual_num_gpus = cuda_toolkit_8.QueryNumberOfGpus(vm)
  if actual_num_gpus != expected_num_gpus:
    raise Exception('VM reported incorrect number of GPUs. ',
                    'Expected %s, received %s' %
                    (expected_num_gpus, actual_num_gpus))


def _InstallAndAuthenticateVm(vm):
    vm.Install('shoc_benchmark_suite')
    AssertCorrectNumberOfGpus(vm)
    cuda_toolkit_8.SetAndConfirmGpuClocks(vm)
    vm.AuthenticateVm()  # Configure ssh between vms for MPI


def Prepare(benchmark_spec):
  """Install SHOC.

  Args:
    benchmark_spec: The benchmark specification. Contains all data that is
        required to run the benchmark.
  """

  vm_util.RunThreaded(_InstallAndAuthenticateVm, benchmark_spec.vms)

  master_vm = benchmark_spec.vms[0]
  num_gpus = cuda_toolkit_8.QueryNumberOfGpus(master_vm)
  CreateAndPushMachineFile(benchmark_spec.vms, num_gpus)


def CreateAndPushMachineFile(vms, num_gpus):
  """Create a file with the IP of each machine in the cluster on its own line.
     TODO: support different number of gpus per machine
     TODO: add num_gpus to linux_virtual_machine

  Args:
    vms: The list of vms which will be in the cluster.
  """
  with vm_util.NamedTemporaryFile() as machine_file:
    master_vm = vms[0]
    machine_file.write('localhost slots=%d\n' % num_gpus)
    for vm in vms[1:]:
      machine_file.write('%s slots=%d\n' % (vm.internal_ip,
                                            num_gpus))
    machine_file.close()
    master_vm.PushFile(machine_file.name, MACHINEFILE)


def _ExtractResult(shoc_output, result_name):
  result_line = [x for x in shoc_output.splitlines() if x.find(result_name) != -1][0].split() #TODO: ew
  result_value = float(result_line[-2])
  result_units = result_line[-1]
  return (result_value, result_units)


def _MakeSamplesFromOutput(stdout, metadata):
  results = []
  for metric in ('stencil:', 'stencil_dp:'):
    (value, unit) = _ExtractResult(stdout, metric)
    results.append(sample.Sample(
        metric[:-1], # strip trailing colon
        value,
        unit,
        metadata))
  return results


def _MakeSamplesFromStencilOutput(stdout, metadata):
  dp_median_results = [x for x in stdout.splitlines() if x.find('DP_Sten2D(median)') != -1][0].split()
  dp_units = dp_median_results[2]
  dp_median = float(dp_median_results[3])

  dp_stddev_results = [x for x in stdout.splitlines() if x.find('DP_Sten2D(stddev)') != -1][0].split()
  dp_stddev_units = dp_stddev_results[2]
  dp_stddev_median = float(dp_stddev_results[3])

  sp_median_results = [x for x in stdout.splitlines() if x.find('SP_Sten2D(median)') != -1][0].split()
  sp_units = sp_median_results[2]
  sp_median = float(sp_median_results[3])

  sp_stddev_results = [x for x in stdout.splitlines() if x.find('SP_Sten2D(stddev)') != -1][0].split()
  sp_stddev_units = sp_stddev_results[2]
  sp_stddev_median = float(sp_stddev_results[3])

  results = []
  results.append(sample.Sample(
      'Stencil2D DP median',
      dp_median,
      dp_units,
      metadata))

  results.append(sample.Sample(
      'Stencil2D DP stddev',
      dp_stddev_median,
      dp_stddev_units,
      metadata))

  results.append(sample.Sample(
      'Stencil2D SP median',
      sp_median,
      sp_units,
      metadata))

  results.append(sample.Sample(
      'Stencil2D SP stddev',
      sp_stddev_median,
      sp_stddev_units,
      metadata))

  return results


def Run(benchmark_spec):
  """Sets the GPU clock speed and runs the SHOC benchmark.

  Args:
    benchmark_spec: The benchmark specification. Contains all data that is
        required to run the benchmark.

  Returns:
    A list of sample.Sample objects.
  """
  vms = benchmark_spec.vms
  master_vm = vms[0]
  num_gpus = cuda_toolkit_8.QueryNumberOfGpus(master_vm)  #TODO: Dont replicate
  num_iterations = FLAGS.shoc_iterations
  #problem_size = FLAGS.problem_size
  stencil2d_path = os.path.join(shoc_benchmark_suite.SHOC_BIN_DIR,
                                'TP', 'CUDA', 'Stencil2D')
  num_processes = len(vms) * num_gpus

  metadata = {}
  metadata['benchmark_version'] = BENCHMARK_VERSION
  metadata['num_iterations'] = num_iterations
  metadata['gpus_per_node'] = num_gpus
  metadata['memory_clock_MHz'] = FLAGS.gpu_clock_speeds[0]
  metadata['graphics_clock_MHz'] = FLAGS.gpu_clock_speeds[1]
  metadata['num_nodes'] = len(vms)
  metadata['num_processes'] = num_processes

  results = []

  initial_problem_size = 1024
  max_problem_size = 20000
  problem_size_step = 1024

  for problem_size in range(initial_problem_size, max_problem_size + 1, problem_size_step):
    current_problem_size = '%s,%s' % (problem_size, problem_size)

    run_command = ('mpirun --hostfile %s -np %s %s --customSize %s -n %s' %
                   (MACHINEFILE, num_processes, stencil2d_path, current_problem_size,
                    num_iterations))

    metadata['run_command'] = run_command
    metadata['problem_size'] = current_problem_size

    stdout, _ = master_vm.RemoteCommand(run_command, should_log=True)
    results.extend(_MakeSamplesFromStencilOutput(stdout, metadata.copy()))

  return results


def Cleanup(benchmark_spec):
  pass

