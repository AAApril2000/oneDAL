/* file: Step3LocalCollectionInputId.java */
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
 * @ingroup dbscan_compute
 * @{
 */
package com.intel.daal.algorithms.dbscan;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__STEP3LOCALCOLLECTIONINPUTID"></a>
 * @brief Available identifiers of input data collection objects for the DBSCAN algorithm in the third step
 *        of the distributed processing mode
 */
public final class Step3LocalCollectionInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public Step3LocalCollectionInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step3PartialBoundingBoxesValue = 2;

    public static final Step3LocalCollectionInputId step3PartialBoundingBoxes = new Step3LocalCollectionInputId(step3PartialBoundingBoxesValue);
       /*!< Collection of input tables containing bounind boxes */
}
/** @} */
