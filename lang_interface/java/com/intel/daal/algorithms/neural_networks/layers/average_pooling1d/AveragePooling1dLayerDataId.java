/* file: AveragePooling1dLayerDataId.java */
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
 * @ingroup average_pooling1d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.average_pooling1d;

import java.lang.annotation.Native;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__AVERAGE_POOLING1D__AVERAGEPOOLING1DLAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward one-dimensional average pooling layer
 *        and results for the forward one-dimensional average pooling layer
 */
public final class AveragePooling1dLayerDataId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public AveragePooling1dLayerDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int auxInputDimensionsId = 0;

    public static final AveragePooling1dLayerDataId auxInputDimensions = new AveragePooling1dLayerDataId(
        auxInputDimensionsId);    /*!< Numeric table that stores forward average pooling layer results */
}
/** @} */
