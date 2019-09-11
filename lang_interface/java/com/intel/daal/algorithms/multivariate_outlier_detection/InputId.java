/* file: InputId.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @ingroup multivariate_outlier_detection
 * @{
 */
package com.intel.daal.algorithms.multivariate_outlier_detection;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__INPUTID"></a>
 * @brief Available identifiers of input objects for the multivariate outlier detection algorithm
 */
public final class InputId {

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public InputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int dataValue = 0;
    private static final int locationValue = 1;
    private static final int scatterValue = 2;
    private static final int thresholdValue = 3;

    /** Input data table */
    public static final InputId data = new InputId(dataValue);
    public static final InputId location = new InputId(locationValue);
    public static final InputId scatter = new InputId(scatterValue);
    public static final InputId threshold = new InputId(thresholdValue);
}
/** @} */
