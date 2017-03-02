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


flags.DEFINE_integer('shoc_iterations', 1,
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
BENCHMARK_VERSION = '0.23'
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

def RunCmd(vm, command, user=None, command_context=None):
  if command_context:
    command = command.format(**command_context)
  command = MakeRemoteCommand(user, command)
  return vm.RemoteCommand(command)


def MakeRemoteCommand(user, command):
  if user is None:
    return command
  else:
    return "sudo -u {user} -i -- sh -c '{command}'".format(
        user=user, command=command)


def ClearNetworking(vms):
  cmds = [NETWORK_DELAY_TEARDOWN_CMD,]
  cmds.append(NETWORK_BUSY_READ.format(0))
  cmds.append(NETWORK_BUSY_POLL.format(0))
  cmd = ';'.join(cmds)
  vm_util.RunThreaded(lambda vm: RunCmd(vm, cmd, 'root'), vms)


def SetNetworkSettings(vms, busy_read, busy_poll, network_delay):
  """Sets network features if non zero, returning options set.

  Args:
    vms: worker vms to run on
    busy_read: integer to set busy read to
    busy_poll: integer to set busy poll to
    network_delay: float of milliseconds for ethernet delay

  Returns:
    Dict of non-zero network settings for putting into labels
  """

  cmds = list()
  context = dict()
  if busy_poll:
    cmds.append(NETWORK_BUSY_POLL.format(busy_poll))
    context['network_busy_poll'] = busy_poll
  if busy_read:
    cmds.append(NETWORK_BUSY_READ.format(busy_read))
    context['network_busy_read'] = busy_read
  if network_delay:
    cmds.append(NETWORK_DELAY_CMD.format(network_delay))
    context['network_delay_ms'] = network_delay
  cmd = ';'.join(cmds)
  if cmd:
    vm_util.RunThreaded(lambda vm: RunCmd(vm, cmd, 'root'), vms)
  return context


def Prepare(benchmark_spec):
  """Install SHOC.

  Args:
    benchmark_spec: The benchmark specification. Contains all data that is
        required to run the benchmark.
  """
  for vm in benchmark_spec.vms:  #TODO: run-threaded
    vm.Install('shoc_benchmark_suite')
    AssertCorrectNumberOfGpus(vm)
    cuda_toolkit_8.SetAndConfirmGpuClocks(vm)
    vm.AuthenticateVm()  # Configure ssh between vms for MPI

  master_vm = benchmark_spec.vms[0]
  num_gpus = cuda_toolkit_8.QueryNumberOfGpus(master_vm)
  CreateAndPushMachineFile(benchmark_spec.vms, num_gpus)
  #busy_read = 0
  #busy_poll = 0
  #network_delay = 5
  #SetNetworkSettings(benchmark_spec.vms, busy_read, busy_poll, network_delay)


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
  dp_mean_results = [x for x in stdout.splitlines() if x.find('DP_Sten2D(mean)') != -1][0].split()
  dp_units = dp_mean_results[2]
  dp_median = float(dp_mean_results[3])
  dp_mean = float(dp_mean_results[4])
  dp_stddev = float(dp_mean_results[5])
  dp_min = float(dp_mean_results[6])
  dp_max = float(dp_mean_results[7])

  sp_mean_results = [x for x in stdout.splitlines() if x.find('SP_Sten2D(mean)') != -1][0].split()
  sp_units = sp_mean_results[2]
  sp_median = float(sp_mean_results[3])
  sp_mean = float(sp_mean_results[4])
  sp_stddev = float(sp_mean_results[5])
  sp_min = float(sp_mean_results[6])
  sp_max = float(sp_mean_results[7])

  results = []
  results.append(sample.Sample(
      'Stencil2D DP mean',
      dp_mean,
      dp_units,
      metadata))

  results.append(sample.Sample(
      'Stencil2D SP mean',
      sp_mean,
      sp_units,
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
  problem_size = FLAGS.problem_size
  stencil2d_path = os.path.join(shoc_benchmark_suite.SHOC_BIN_DIR,
                                'TP', 'CUDA', 'Stencil2D')
  num_processes = len(vms) * num_gpus
  run_command = ('mpirun --hostfile %s -np %s %s --customSize %s' %
                 (MACHINEFILE, num_processes, stencil2d_path, problem_size))
  metadata = {}
  results = []
  metadata['benchmark_version'] = BENCHMARK_VERSION
  metadata['num_iterations'] = num_iterations
  metadata['gpus_per_node'] = num_gpus
  metadata['memory_clock_MHz'] = FLAGS.gpu_clock_speeds[0]
  metadata['graphics_clock_MHz'] = FLAGS.gpu_clock_speeds[1]
  metadata['run_command'] = run_command
  metadata['num_nodes'] = len(vms)
  metadata['num_processes'] = num_processes
  metadata['problem_size'] = problem_size

  stdout, _ = master_vm.RemoteCommand(run_command, should_log=True)
  results.extend(_MakeSamplesFromStencilOutput(stdout, metadata))
  return results


def Cleanup(benchmark_spec):
  pass

