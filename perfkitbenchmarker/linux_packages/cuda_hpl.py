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


"""Module containing NVIDIA's CUDA Accelerated Linpack installation and 
   cleanup functions. 

   Due to NVIDIA's license, the tarball must be downloaded
   manually from NVIDIA and placed in PKB's data folder.

The package can be downloaded here (registration required):
https://developer.nvidia.com/accelerated-computing-developer-program-home
"""

import re

#from perfkitbenchmarker.linux_packages import openblas
#from perfkitbenchmarker.linux_packages import cuda_toolkit_8
from perfkitbenchmarker.linux_packages import INSTALL_DIR
from perfkitbenchmarker import data

HPL_TAR = 'hpl-2.0_FERMI_v15.tgz'
HPL_PATCH = 'hpl_cuda.patch'
HPL_DIR = '%s/hpl-2.0_FERMI_v15' % INSTALL_DIR
MAKE_FLAVOR = 'Linux_PII_CBLAS'
#HPCC_MAKEFILE = 'Make.' + MAKE_FLAVOR
#HPCC_MAKEFILE_PATH = HPCC_DIR + '/hpl/' + HPCC_MAKEFILE


def _Install(vm):
  """Installs the CUDA HPL package on the VM."""
  vm.Install('wget')
  #vm.Install('openmpi')
  #vm.Install('openblas')
  vm.InstallPackages('libopenblas-dev libopenmpi-dev') #TODO: no
  vm.Install('cuda_toolkit_8')
  vm.PushFile(data.ResourcePath(HPL_TAR), INSTALL_DIR)
  vm.PushFile(data.ResourcePath(HPL_PATCH), INSTALL_DIR)
  vm.RemoteCommand('cd %s && tar xvf %s' % (INSTALL_DIR, HPL_TAR))
  vm.RemoteCommand('cd %s && patch -p0 < %s' % (INSTALL_DIR, HPL_PATCH))
  sed_cmd = 'sed -i s,HPL_DIR=.*$,HPL_DIR=%s, bin/CUDA/run_linpack' % HPL_DIR
  vm.RemoteCommand('cd %s && %s' % (INSTALL_DIR, sed_cmd))
  #vm.RemoteCommand(
  #    'cp %s/hpl/setup/%s %s' % (HPCC_DIR, HPCC_MAKEFILE, HPCC_MAKEFILE_PATH))
  #sed_cmd = (
  #    'sed -i -e "/^MP/d" -e "s/gcc/mpicc/" -e "s/g77/mpicc/" '
  #    '-e "s/\\$(HOME)\\/netlib\\/ARCHIVES\\/Linux_PII/%s/" '
  #    '-e "s/libcblas.*/libopenblas.a/" '
  #    '-e "s/\\-lm/\\-lgfortran \\-lm/" %s' %
  #    (re.escape(openblas.OPENBLAS_DIR), HPCC_MAKEFILE_PATH))
  #vm.RemoteCommand(sed_cmd)
  #vm.RemoteCommand('cd %s; make arch=Linux_PII_CBLAS' % HPCC_DIR)


def YumInstall(vm):
  """Installs the CUDA HPL package on the VM."""
  _Install(vm)


def AptInstall(vm):
  """Installs the CUDA HPL package on the VM."""
  _Install(vm)
