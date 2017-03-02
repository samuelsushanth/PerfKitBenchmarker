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

"""Tests for SHOC benchmark suite benchmark."""
import os
import unittest

import mock

from perfkitbenchmarker.linux_benchmarks import shoc_benchmark


class ShocBenchmarkTestCase(unittest.TestCase):

  def setUp(self):
    p = mock.patch(shoc_benchmark.__name__ + '.FLAGS')
    p.start()
    self.addCleanup(p.stop)

    path = os.path.join(os.path.dirname(__file__), '../data',
                        'stencil2d_output.txt')
    with open(path) as fp:
      self.test_output = fp.read()

  @unittest.skip('old')
  def testMakeSampleFromOutput(self):
    testMetadata = { 'foo': 'bar' }
    actual = shoc_benchmark._MakeSamplesFromOutput(
        self.test_output, testMetadata)
    results_dict = { x.metric: x for x in actual }
    stencil_results = results_dict['stencil']
    self.assertEqual('stencil', stencil_results.metric)
    self.assertEqual(124.8580, stencil_results.value)
    self.assertEqual('GFLOPS', stencil_results.unit)
    self.assertEqual(testMetadata, stencil_results.metadata)

  def testMakeSampleFromOutput(self):
    testMetadata = { 'foo': 'bar' }
    actual = shoc_benchmark._MakeSamplesFromStencilOutput(
        self.test_output, testMetadata)
    results_dict = { x.metric: x for x in actual }

    stencil_dp_results = results_dict['Stencil2D DP mean']
    self.assertEqual('Stencil2D DP mean', stencil_dp_results.metric)
    self.assertEqual(474.128, stencil_dp_results.value)
    self.assertEqual('GFLOPS', stencil_dp_results.unit)
    self.assertEqual(testMetadata, stencil_dp_results.metadata)

    stencil_sp_results = results_dict['Stencil2D SP mean']
    self.assertEqual('Stencil2D SP mean', stencil_sp_results.metric)
    self.assertEqual(750.593, stencil_sp_results.value)
    self.assertEqual('GFLOPS', stencil_sp_results.unit)
    self.assertEqual(testMetadata, stencil_sp_results.metadata)

if __name__ == '__main__':
  unittest.main()

