/* file: predict_batch.cpp */
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

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>

#include "daal.h"
#include "com_intel_daal_algorithms_decision_forest_classification_prediction_PredictionBatch.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
namespace dfcp = daal::algorithms::decision_forest::classification::prediction;

/*
* Class:     com_intel_daal_algorithms_decision_forest_classification_prediction_PredictionBatch
* Method:    cInit
* Signature: (IIJ)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1forest_classification_prediction_PredictionBatch_cInit
(JNIEnv *, jobject thisObj, jint prec, jint method, jlong nClasses)
{
    return jniBatch<dfcp::Method, dfcp::Batch, dfcp::defaultDense>::newObj(prec, method, nClasses);
}

/*
* Class:     com_intel_daal_algorithms_decision_forest_classification_prediction_PredictionBatch
* Method:    cInitParameter
* Signature: (JIII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1forest_classification_prediction_PredictionBatch_cInitParameter
(JNIEnv *, jobject thisObj, jlong algAddr, jint prec, jint method, jint cmode)
{
    return jniBatch<dfcp::Method, dfcp::Batch, dfcp::defaultDense>::getParameter(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_decision_forest_classification_prediction_PredictionBatch
* Method:    cClone
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1forest_classification_prediction_PredictionBatch_cClone
(JNIEnv *, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<dfcp::Method, dfcp::Batch, dfcp::defaultDense>::getClone(prec, method, algAddr);
}
