/* file: train_input.cpp */
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
#include "com_intel_daal_algorithms_decision_tree_regression_training_TrainingInput.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()

namespace dtrt = daal::algorithms::decision_tree::regression::training;

/*
* Class:     com_intel_daal_algorithms_decision_tree_regression_training_TrainingInput
* Method:    cSetInput
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_decision_1tree_regression_training_TrainingInput_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<dtrt::Input>::set<dtrt::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_regression_training_TrainingInput
* Method:    cGetInput
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1tree_regression_training_TrainingInput_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<dtrt::Input>::get<dtrt::InputId, NumericTable>(inputAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_regression_training_TrainingInput
* Method:    cGetNumberOfFeatures
* Signature: (J)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1tree_regression_training_TrainingInput_cGetNumberOfFeatures
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
    return ((dtrt::Input*)inputAddr)->getNumberOfFeatures();
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_regression_training_TrainingInput
* Method:    cGetNumberOfDependentVariables
* Signature: (J)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1tree_regression_training_TrainingInput_cGetNumberOfDependentVariables
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
    return ((dtrt::Input*)inputAddr)->getNumberOfDependentVariables();
}
