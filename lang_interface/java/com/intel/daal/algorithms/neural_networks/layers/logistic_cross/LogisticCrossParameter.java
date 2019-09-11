/* file: LogisticCrossParameter.java */
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
 * @ingroup logistic_cross
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.logistic_cross;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOGISTIC_CROSS__LOGISTICCROSSPARAMETER"></a>
 * \brief Class that specifies parameters of the logistic cross-entropy layer
 */
public class LogisticCrossParameter extends com.intel.daal.algorithms.neural_networks.layers.loss.LossParameter {

    /**
     *  Constructs the parameters for the logistic cross-entropy layer
     */
    public LogisticCrossParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public LogisticCrossParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    private native long   cInit();
}
/** @} */
