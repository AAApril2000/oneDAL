/* file: input.cpp */
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
#include <jni.h>/* Header for class com_intel_daal_algorithms_association_rules_Batch */

#include "daal.h"
#include "com_intel_daal_algorithms_association_rules_Input.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_association_1rules_Input_cInit
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jint cmode)
{
    return jniBatch<association_rules::Method, association_rules::Batch, association_rules::apriori>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_association_rules_Input
 * Method:    cSetDataSet
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_association_1rules_Input_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<association_rules::Input>::set<association_rules::InputId, NumericTable>(inputAddr, id, ntAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_association_1rules_Input_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<association_rules::Input>::get<association_rules::InputId, NumericTable>(inputAddr, id);
}
