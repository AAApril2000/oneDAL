/* file: distributed_step4_local_input.cpp */
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
#include "com_intel_daal_algorithms_dbscan_DistributedStep4LocalInput.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::dbscan;

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep4LocalInput
* Method:    cSetDataCollection
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep4LocalInput_cSetDataCollection
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong dcAddr)
{
    if (id == partialData || id == partialWeights)
    {
        jniInput<DistributedInput<step4Local> >::set<LocalCollectionInputId, DataCollection>(inputAddr, id, dcAddr);
    }
    else
    {
        jniInput<DistributedInput<step4Local> >::set<Step4LocalCollectionInputId, DataCollection>(inputAddr, id, dcAddr);
    }
}

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep4LocalInput
* Method:    cAddNumericTable
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep4LocalInput_cAddNumericTable
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == partialData || id == partialWeights)
    {
        jniInput<DistributedInput<step4Local> >::add<LocalCollectionInputId, NumericTable>(inputAddr, id, ntAddr);
    }
    else
    {
        jniInput<DistributedInput<step4Local> >::add<Step4LocalCollectionInputId, NumericTable>(inputAddr, id, ntAddr);
    }
}

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep4LocalInput
* Method:    cGetDataCollection
* Signature:(JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep4LocalInput_cGetDataCollection
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    if (id == partialData || id == partialWeights)
    {
        return jniInput<DistributedInput<step4Local> >::get<LocalCollectionInputId, DataCollection>(inputAddr, id);
    }
    else
    {
        return jniInput<DistributedInput<step4Local> >::get<Step4LocalCollectionInputId, DataCollection>(inputAddr, id);
    }
}
