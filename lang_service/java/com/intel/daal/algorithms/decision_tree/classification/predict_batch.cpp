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
#include "com_intel_daal_algorithms_decision_tree_classification_prediction_PredictionBatch.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()

namespace dtcp = daal::algorithms::decision_tree::classification::prediction;

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_prediction_PredictionBatch
* Method:    cInit
* Signature: (IIJ)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_prediction_PredictionBatch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<dtcp::Method, dtcp::Batch, dtcp::defaultDense>::newObj(prec, method);
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_prediction_PredictionBatch
* Method:    cInitParameter
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_prediction_PredictionBatch_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<dtcp::Method, dtcp::Batch, dtcp::defaultDense>::getParameter(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_prediction_PredictionBatch
* Method:    cInitParameter
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_prediction_PredictionBatch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<dtcp::Method, dtcp::Batch, dtcp::defaultDense>::getInput(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_prediction_PredictionBatch
* Method:    cClone
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_prediction_PredictionBatch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<dtcp::Method, dtcp::Batch, dtcp::defaultDense>::getClone(prec, method, algAddr);
}
