/* file: training_init_input.cpp */
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

#include <jni.h>

#include "daal.h"

#include "com_intel_daal_algorithms_implicit_als_training_init_InitInput.h"

#include "implicit_als_init_defines.i"

#include "common_helpers.h"
#include "common_defines.i"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::training::init;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitInput
 * Method:    cInit
 * Signature: (JIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitInput_cInit
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jint cmode)
{
    if (cmode == jBatch)
    {
        return jniBatch<implicit_als::training::init::Method, Batch, fastCSR, defaultDense>::getInput(prec, method, algAddr);
    }
    else if (cmode == jDistributed)
    {
        return jniDistributed<step1Local, implicit_als::training::init::Method, Distributed, fastCSR, defaultDense>::getInput(prec, method, algAddr);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitInput_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<implicit_als::training::init::Input>::set<InputId, NumericTable>(inputAddr, data, ntAddr);
}


/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitInput
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitInput_cGetInputTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)

{
    return jniInput<implicit_als::training::init::Input>::get<InputId, NumericTable>(inputAddr, data);
}
