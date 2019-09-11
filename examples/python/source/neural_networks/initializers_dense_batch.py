# file: initializers_dense_batch.py
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
# !    Python example of initializers
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-INITIALIZERS_DENSE_BATCH"></a>
## \example initializers_dense_batch.py
#

import os
import sys

from daal.algorithms.neural_networks import layers
from daal.algorithms.neural_networks import initializers
from daal.data_management import HomogenTensor, TensorIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensor

if __name__ == "__main__":
    # Create collection of dimension sizes of the input data tensor
    inDims = [2, 1, 3, 4]
    tensorData = HomogenTensor(inDims, TensorIface.doAllocate)

    # Fill tensor data using truncated gaussian initializer
    # Create an algorithm to initialize data using default method
    truncatedGaussInitializer = initializers.truncated_gaussian.Batch(0.0, 1.0)

    # Set input object and parameters for the truncated gaussian initializer
    truncatedGaussInitializer.input.set(initializers.data, tensorData)

    # Compute truncated gaussian initializer
    truncatedGaussInitializer.compute()

    # Print the results of the truncated gaussian initializer
    printTensor(tensorData, "Data with truncated gaussian distribution:")


    # Fill tensor data using gaussian initializer
    # Create an algorithm to initialize data using default method
    gaussInitializer = initializers.gaussian.Batch(1.0, 0.5)

    # Set input object and parameters for the gaussian initializer
    gaussInitializer.input.set(initializers.data, tensorData)

    # Compute gaussian initializer
    gaussInitializer.compute()

    # Print the results of the gaussian initializer
    printTensor(tensorData, "Data with gaussian distribution:")


    # Fill tensor data using uniform initializer
    # Create an algorithm to initialize data using default method
    uniformInitializer = initializers.uniform.Batch(-5.0, 5.0)

    # Set input object and parameters for the uniform initializer
    uniformInitializer.input.set(initializers.data, tensorData)

    # Compute uniform initializer
    uniformInitializer.compute()

    # Print the results of the uniform initializer
    printTensor(tensorData, "Data with uniform distribution:")


    # Fill layer weights using xavier initializer
    # Create an algorithm to compute forward fully-connected layer results using default method
    fullyconnectedLayerForward = layers.fullyconnected.forward.Batch(5)

    # Set input objects and parameter for the forward fully-connected layer
    fullyconnectedLayerForward.input.setInput(layers.forward.data, tensorData)
    fullyconnectedLayerForward.parameter.weightsInitializer = initializers.xavier.Batch()

    # Compute forward fully-connected layer results
    fullyconnectedLayerForward.compute()

    # Print the results of the xavier initializer
    printTensor(fullyconnectedLayerForward.input.getInput(layers.forward.weights), "Weights filled by xavier initializer:")
