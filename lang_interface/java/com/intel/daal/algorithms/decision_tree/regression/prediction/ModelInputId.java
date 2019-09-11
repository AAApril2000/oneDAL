/* file: ModelInputId.java */
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
 * @defgroup decision_tree_regression_prediction Prediction
 * @brief Contains classes for making prediction based on the decision_tree regression model
 * @ingroup decision_tree_regression
 * @{
 */
/**
 * @brief Contains classes for making prediction based on the decision_tree regression model
 */
package com.intel.daal.algorithms.decision_tree.regression.prediction;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__REGRESSION__PREDICTION__MODELINPUTID"></a>
 * @brief Available identifiers of input models for making decision tree model-based prediction
 */
public final class ModelInputId {
    private int _value;

    /**
     * Constructs the model input object identifier using the provided value
     * @param value     Value corresponding to the model input object identifier
     */
    public ModelInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the model input object identifier
     * @return Value corresponding to the model input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int modelId = 1;

    public static final ModelInputId model = new ModelInputId(modelId); /*!< Trained decision tree model */
}
/** @} */
