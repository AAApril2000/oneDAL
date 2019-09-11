# file: out_detect_uni_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-OUTLIER_DETECTION_UNIVARIATE_BATCH"></a>
## \example out_detect_uni_dense_batch.py

import os
import sys

from daal.algorithms import univariate_outlier_detection
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
datasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'outlierdetection.csv')

if __name__ == "__main__":

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file
    dataSource = FileDataSource(
        datasetFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the data from the input file
    dataSource.loadDataBlock()

    nFeatures = dataSource.getNumberOfColumns()

    algorithm = univariate_outlier_detection.Batch()

    algorithm.input.set(univariate_outlier_detection.data, dataSource.getNumericTable())

    # Compute outliers and get the computed results
    res = algorithm.compute()

    printNumericTable(dataSource.getNumericTable(), "Input data")
    printNumericTable(res.get(univariate_outlier_detection.weights), "Outlier detection result (univariate)")
