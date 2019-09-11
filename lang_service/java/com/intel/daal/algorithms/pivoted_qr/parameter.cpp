/* file: parameter.cpp */
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
#include <jni.h>/* Header for class com_intel_daal_algorithms_pivoted_qr_Offline */
#include "pivoted_qr_types.i"

#include "daal.h"
#include "com_intel_daal_algorithms_pivoted_qr_Parameter.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::services;

/*
 * Class:     com_intel_daal_algorithms_pivoted_qr_Parameter
 * Method:    cSetPermutedColumns
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pivoted_1qr_Parameter_cSetPermutedColumns
(JNIEnv *env, jobject thisObj, jlong parAddr, jlong permutedColumnsAddr)
{
    SerializationIfacePtr *ntShPtr = (SerializationIfacePtr *)permutedColumnsAddr;
    (*(pivoted_qr::Parameter *)parAddr).permutedColumns = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_pivoted_qr_Parameter
 * Method:    cGetPermutedColumns
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pivoted_1qr_Parameter_cGetPermutedColumns
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    pivoted_qr::Parameter *parameter = (pivoted_qr::Parameter *)parAddr;
    NumericTablePtr *ntShPtr = new NumericTablePtr();
    *ntShPtr = parameter->permutedColumns;
    return (jlong)ntShPtr;
}
