/* file: AveragePooling2dBatch.java */
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
 * @defgroup average_pooling2d Two-dimensional Average Pooling Layer
 * @brief Contains classes for average two-dimensional (2D) pooling layer
 * @ingroup pooling2d
 * @{
 */
/**
 * @brief Contains classes of the two-dimensional (2D) average pooling layer
 */
package com.intel.daal.algorithms.neural_networks.layers.average_pooling2d;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__AVERAGE_POOLING2D__AVERAGEPOOLING2DBATCH"></a>
 * @brief Provides methods for the two-dimensional average pooling layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-AVERAGEPOOLING2DFORWARD-ALGORITHM">Forward two-dimensional average pooling layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-AVERAGEPOOLING2DBACKWARD-ALGORITHM">Backward two-dimensional average pooling layer description and usage models</a> -->
 *
 * @par References
 *      - @ref AveragePooling2dForwardBatch class
 *      - @ref AveragePooling2dBackwardBatch class
 */
public class AveragePooling2dBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public    AveragePooling2dMethod        method;      /*!< Computation method for the layer */
    public    AveragePooling2dParameter     parameter;   /*!< Pooling layer parameters */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the two-dimensional average pooling layer
     * @param context    Context to manage the two-dimensional average pooling layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref AveragePooling2dMethod
     * @param nDim       Number of dimensions in input data
     */
    public AveragePooling2dBatch(DaalContext context, Class<? extends Number> cls, AveragePooling2dMethod method, long nDim) {
        super(context);

        this.method = method;

        if (method != AveragePooling2dMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue(), nDim);
        parameter = new AveragePooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));

        forwardLayer = (ForwardLayer)(new AveragePooling2dForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue()), nDim));
        backwardLayer = (BackwardLayer)(new AveragePooling2dBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(),
                                                                                                  method.getValue()), nDim));
    }

    private native long cInit(int prec, int method, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
