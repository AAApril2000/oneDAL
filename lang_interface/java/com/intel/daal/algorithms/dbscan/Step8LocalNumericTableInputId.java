/* file: Step8LocalNumericTableInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__STEP8LOCALNUMERICTABLEINPUTID"></a>
 * @brief Available identifiers of input numeric table objects for the DBSCAN algorithm in the eighth step
 *        of the distributed processing mode
 */
public final class Step8LocalNumericTableInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public Step8LocalNumericTableInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step8InputClusterStructureValue = 0;
    private static final int step8InputNClustersValue        = 1;

    public static final Step8LocalNumericTableInputId step8InputClusterStructure = new Step8LocalNumericTableInputId(step8InputClusterStructureValue);
       /*!< Input table containing information about current clustering state of observations */
    public static final Step8LocalNumericTableInputId step8InputNClusters = new Step8LocalNumericTableInputId(step8InputNClustersValue);
       /*!< Input table containing the current number of clusters */
}
/** @} */
