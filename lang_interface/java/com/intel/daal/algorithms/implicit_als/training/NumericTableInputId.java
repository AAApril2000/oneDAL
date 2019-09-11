/* file: NumericTableInputId.java */
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
 * @ingroup implicit_als_training_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training;

import java.lang.annotation.Native;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__NUMERICTABLEINPUTID"></a>
 * @brief Available identifiers of input numeric table objects for the implicit ALS
 * training algorithm in the distributed processing mode
 */
public final class NumericTableInputId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the input numeric table object identifier using the provided value
     * @param value     Value corresponding to the input numeric table object identifier
     */
    public NumericTableInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input numeric table object identifier
     * @return Value corresponding to the input numeric table object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int dataId = 0;

    /** %Input data table */
    public static final NumericTableInputId data = new NumericTableInputId(
            dataId); /*!< %Input data table that contains ratings */
}
/** @} */
