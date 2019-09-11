/* file: Parameter.java */
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
 * @ingroup iterative_solver
 * @{
 */
/**
 * @brief Contains classes for computing iterative solver algorithm
 */
package com.intel.daal.algorithms.optimization_solver.iterative_solver;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.algorithms.optimization_solver.sum_of_functions.Batch;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__PARAMETER"></a>
 * @brief Parameter of the iterative solver algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the parameter for the iterative solver algorithm
     * @param context       Context to manage the parameter for the iterative solver algorithm
     */
    public Parameter(DaalContext context) {
        super(context);
    }

    /**
     * Constructs the parameter for the iterative solver algorithm
     * @param context    Context to manage the iterative solver algorithm
     * @param cObject    Pointer to C++ implementation of the parameter
     */
    public Parameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
    * Sets objective function represented as sum of functions
    * @param function Objective function represented as sum of functions
    */
    public void setFunction(Batch function) {
        _function = function;
        cSetFunction(this.cObject, function.cBatchIface);
    }

    /**
     * Gets objective function represented as sum of functions
     * @return Objective function represented as sum of functions
     */
    public Batch getFunction() {
        return _function;
    }

    /**
    * Sets the maximal number of iterations of the algorithm
    * @param nIterations The maximal number of iterations of the algorithm
    */
    public void setNIterations(long nIterations) {
        cSetNIterations(this.cObject, nIterations);
    }

    /**
     * Gets the maximal number of iterations of the algorithm
     * @return The maximal number of iterations of the algorithm
     */
    public long getNIterations() {
        return cGetNIterations(this.cObject);
    }

    /**
    * Sets the accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
    * @param accuracyThreshold The accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
    */
    public void setAccuracyThreshold(double accuracyThreshold) {
        cSetAccuracyThreshold(this.cObject, accuracyThreshold);
    }

    /**
     * Gets the accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * @return The accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     */
    public double getAccuracyThreshold() {
        return cGetAccuracyThreshold(this.cObject);
    }

    /**
     * Sets the optionalResultRequired flag
     * @param flag    The flag. If true, optional result is calculated
     */
    public void setOptionalResultRequired(boolean flag) {
        cSetOptionalResultRequired(this.cObject, flag);
    }

    /**
     * Gets the optionalResultRequired flag
     * @return The flag
     */
    public boolean getOptionalResultRequired() {
        return cGetOptionalResultRequired(this.cObject);
    }

    /**
    * Sets the number of batch indices to compute the stochastic gradient.
    * If batchSize is equal to the number of terms in objective
    * function then no random sampling is performed, and all terms are
    * used to calculate the gradient. This parameter is ignored
    * if batchIndices is provided.
    * @param batchSize The number of batch indices to compute the stochastic gradient.
    * If batchSize is equal to the number of terms in objective
    * function then no random sampling is performed, and all terms are
    * used to calculate the gradient. This parameter is ignored
    * if batchIndices is provided.
    */
    public void setBatchSize(long batchSize) {
        cSetBatchSize(this.cObject, batchSize);
    }

    /**
    * Returns the number of batch indices to compute the stochastic gradient.
    * If batchSize is equal to the number of terms in objective
    * function then no random sampling is performed, and all terms are
    * used to calculate the gradient. This parameter is ignored
    * if batchIndices is provided.
    * @return The number of batch indices to compute the stochastic gradient.
    * If batchSize is equal to the number of terms in objective
    * function then no random sampling is performed, and all terms are
    * used to calculate the gradient. This parameter is ignored
    * if batchIndices is provided.
    */
    public long getBatchSize() {
        return cGetBatchSize(this.cObject);
    }

    private Batch _function;

    private native void cSetFunction(long parAddr, long function);

    private native void cSetNIterations(long parAddr, long nIterations);
    private native long cGetNIterations(long parAddr);

    private native void cSetAccuracyThreshold(long parAddr, double accuracyThreshold);
    private native double cGetAccuracyThreshold(long parAddr);

    private native void cSetOptionalResultRequired(long parAddr, boolean flag);
    private native boolean cGetOptionalResultRequired(long parAddr);

    private native void cSetBatchSize(long parAddr, long batchSize);
    private native long cGetBatchSize(long parAddr);

}
/** @} */
