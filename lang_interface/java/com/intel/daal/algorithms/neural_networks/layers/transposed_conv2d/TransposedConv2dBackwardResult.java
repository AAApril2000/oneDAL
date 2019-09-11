/* file: TransposedConv2dBackwardResult.java */
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
 * @ingroup transposed_conv2d_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.transposed_conv2d;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TRANSPOSED_CONV2D__TRANSPOSEDCONV2DBACKWARDRESULT"></a>
 * @brief Provides methods to access results obtained with the compute() method of the backward 2D transposed convolution layer
 */
public class TransposedConv2dBackwardResult extends com.intel.daal.algorithms.neural_networks.layers.BackwardResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward 2D transposed convolution layer result
     * @param context   Context to manage the backward 2D transposed convolution layer result
     */
    public TransposedConv2dBackwardResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public TransposedConv2dBackwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    private native long cNewResult();
}
/** @} */
