/* file: Batch.java */
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
 * @defgroup cholesky Cholesky Decomposition
 * @brief Contains classes for computing Cholesky decomposition
 * @ingroup analysis
 * @{
 */
/**
 * @defgroup cholesky_batch Batch
 * @ingroup cholesky
 * @{
 */
package com.intel.daal.algorithms.cholesky;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CHOLESKY__BATCH"></a>
 * \brief Computes Cholesky decomposition in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-CHOLESKY-ALGORITHM">Cholesky decomposition algorithm description and usage models</a> -->
 *
 * \par References
 *      - @ref InputId class
 *      - @ref ResultId class
 */
public class Batch extends AnalysisBatch {
    public Input          input;     /*!< %Input data */
    public Method     method; /*!< Computation method for the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the Cholesky decomposition algorithm by copying input objects
     * of another Cholesky decomposition algorithm
     * @param context    Context to manage the Cholesky decomposition algorithm
     * @param[in] other  An algorithm to be used as the source to initialize the input objects
     *                   and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(context, cObject, prec, method);
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHM__CHOLESKY__BATCH__BATCH"></a>
     * Constructs the Cholesky decomposition algorithm
     *
     * @param context    Context to manage the Cholesky decomposition algorithm
     * @param cls        Data type to use in intermediate computations for Cholesky decomposition, Double.class or Float.class
     * @param method     Cholesky decomposition computation method, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

        this.method = method;

        if (method != Method.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue());
        input = new Input(context, cObject, prec, method);
    }

    /**
     * Computes Cholesky decomposition
     * @return  Cholesky decomposition results
    */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store results of Cholesky decomposition
     * @param result    Structure to store results of Cholesky decomposition
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated Cholesky decomposition algorithm with a copy of input objects
     * of this Cholesky decomposition algorithm
     * @param context    Context to manage the Cholesky decomposition algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cGetResult(long cAlgorithm, int prec, int method);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
/** @} */
