# file: low_order_moms_csr_distr.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

## <a name="DAAL-EXAMPLE-PY-LOW_ORDER_MOMENTS_CSR_DISTRIBUTED"></a>
## \example low_order_moms_csr_distr.py

import os
import sys

from daal import step1Local, step2Master
from daal.algorithms import low_order_moments

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable, createSparseTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
nBlocks = 4

datasetFileNames = [
    os.path.join(DAAL_PREFIX, 'distributed', 'covcormoments_csr_1.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'covcormoments_csr_2.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'covcormoments_csr_3.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'covcormoments_csr_4.csv')
]

partialResult = [0] * nBlocks
result = None


def computestep1Local(block):

    dataTable = createSparseTable(datasetFileNames[block])

    # Create algorithm objects to compute low order moments in the distributed processing mode using the default method
    algorithm = low_order_moments.Distributed(step1Local, method=low_order_moments.fastCSR)

    # Set input objects for the algorithm
    algorithm.input.set(low_order_moments.data, dataTable)

    # Compute partial low order moments estimates on nodes
    partialResult[block] = algorithm.compute()  # Get the computed partial estimates


def computeOnMasterNode():
    global result

    # Create algorithm objects to compute low order moments in the distributed processing mode using the default method
    algorithm = low_order_moments.Distributed(step2Master, method=low_order_moments.fastCSR)

    # Set input objects for the algorithm
    for i in range(nBlocks):
        algorithm.input.add(low_order_moments.partialResults, partialResult[i])

    # Compute a partial low order moments estimate on the master node from the partial estimates on local nodes
    algorithm.compute()

    # Finalize the result in the distributed processing mode and get the computed low order moments
    result = algorithm.finalizeCompute()


def printResults(res):

    printNumericTable(res.get(low_order_moments.minimum),              "Minimum:")
    printNumericTable(res.get(low_order_moments.maximum),              "Maximum:")
    printNumericTable(res.get(low_order_moments.sum),                  "Sum:")
    printNumericTable(res.get(low_order_moments.sumSquares),           "Sum of squares:")
    printNumericTable(res.get(low_order_moments.sumSquaresCentered),   "Sum of squared difference from the means:")
    printNumericTable(res.get(low_order_moments.mean),                 "Mean:")
    printNumericTable(res.get(low_order_moments.secondOrderRawMoment), "Second order raw moment:")
    printNumericTable(res.get(low_order_moments.variance),             "Variance:")
    printNumericTable(res.get(low_order_moments.standardDeviation),    "Standard deviation:")
    printNumericTable(res.get(low_order_moments.variation),            "Variation:")

if __name__ == "__main__":
    for block in range(nBlocks):
        computestep1Local(block)

    computeOnMasterNode()
    printResults(result)
