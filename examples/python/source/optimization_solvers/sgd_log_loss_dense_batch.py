# file: sgd_log_loss_dense_batch.py
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
# !  Python example of the Stochastic gradient descent algorithm with
# !  logistic loss objective function
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-SGD_LOG_LOSS_BATCH"></a>
##  \example sgd_log_loss_dense_batch.py
#

import os
import sys

import numpy as np

import daal.algorithms.optimization_solver as optimization_solver
import daal.algorithms.optimization_solver.logistic_loss
import daal.algorithms.optimization_solver.sgd
import daal.algorithms.optimization_solver.iterative_solver

from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable, MergedNumericTable, NumericTableIface
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

datasetFileName = os.path.join('..', 'data', 'batch', 'custom.csv')

nIterations = 1000
nFeatures = 4
learningRate = 0.01
accuracyThreshold = 0.02

initialPoint = np.array([[1], [1], [1], [1], [1]], dtype=np.float64)

if __name__ == "__main__":

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = FileDataSource(datasetFileName,
                                DataSourceIface.notAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)

    # Create Numeric Tables for data and values for dependent variable
    data = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    dependentVariables = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(data, dependentVariables)

    # Retrieve the data from the input file
    dataSource.loadDataBlock(mergedData)

    nVectors = data.getNumberOfRows()

    logistic_lossObjectiveFunction = optimization_solver.logistic_loss.Batch(nVectors)
    logistic_lossObjectiveFunction.input.set(optimization_solver.logistic_loss.data, data)
    logistic_lossObjectiveFunction.input.set(optimization_solver.logistic_loss.dependentVariables, dependentVariables)

    # Create objects to compute the Stochastic gradient descent result using the default method
    sgdAlgorithm = optimization_solver.sgd.Batch(logistic_lossObjectiveFunction)

    # Set input objects for the the Stochastic gradient descent algorithm
    sgdAlgorithm.input.setInput(optimization_solver.iterative_solver.inputArgument, HomogenNumericTable(initialPoint))
    sgdAlgorithm.parameter.learningRateSequence = HomogenNumericTable(1, 1, NumericTableIface.doAllocate, learningRate)
    sgdAlgorithm.parameter.nIterations = nIterations
    sgdAlgorithm.parameter.accuracyThreshold = accuracyThreshold

    # Compute the Stochastic gradient descent result
    # Result class from daal.algorithms.optimization_solver.iterative_solver
    res = sgdAlgorithm.compute()

    # Print computed the Stochastic gradient descent result
    printNumericTable(res.getResult(optimization_solver.iterative_solver.minimum), "Minimum:")
    printNumericTable(res.getResult(optimization_solver.iterative_solver.nIterations), "Number of iterations performed:")
