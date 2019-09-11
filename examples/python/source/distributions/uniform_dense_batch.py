# file: uniform_dense_batch.py
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

#
# !  Content:
# !    Python example of uniform distribution
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-UNIFORM_DENSE_BATCH"></a>
## \example uniform_dense_batch.py
#

import os
import sys

import daal.algorithms.distributions as distributions
import daal.algorithms.distributions.uniform as uniform
from daal.algorithms.engines.mt19937 import Batch_Float64DefaultDense_create as create
from daal.data_management import HomogenNumericTable, NumericTableIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

if __name__ == "__main__":
    # Create input table to fill with random numbers
    dataTable = HomogenNumericTable(1, 10, NumericTableIface.doAllocate)

    # Create the algorithm
    uniform = uniform.Batch()

    # Set the algorithm input
    uniform.input.set(distributions.tableToFill, dataTable)

    # Set the Mersenne Twister engine to the distribution
    uniform.parameter.engine = create(777)

    # Perform computations
    uniform.compute()

    # Print the results
    printNumericTable(dataTable, "Uniform distribution output:")
