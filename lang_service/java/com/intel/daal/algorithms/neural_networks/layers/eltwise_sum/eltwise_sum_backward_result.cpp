/* file: eltwise_sum_backward_result.cpp */
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
#include "com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumBackwardResult.h"

#include "daal.h"
#include "common_helpers.h"

using namespace daal;
using namespace daal::algorithms::neural_networks::layers::eltwise_sum;
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumBackwardResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumBackwardResult_cNewResult
  (JNIEnv *env, jobject thisObj)
{
    return jniArgument<backward::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_EltwiseSumBackwardResult
 * Method:    cGetTensor
 * Signature: (JIJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumBackwardResult_cGetTensor
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong index)
{
    return jniArgument<backward::Result>::get<layers::backward::ResultLayerDataId, Tensor>(resAddr, id, (size_t)index);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_EltwiseSumBackwardResult
 * Method:    cSetTensor
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumBackwardResult_cSetTensor
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr, jlong index)
{
    jniArgument<backward::Result>::set<layers::backward::ResultLayerDataId, Tensor>(resAddr, id, ntAddr, (size_t)index);
}
