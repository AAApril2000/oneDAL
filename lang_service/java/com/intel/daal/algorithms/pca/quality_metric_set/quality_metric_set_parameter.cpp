/* file: quality_metric_set_parameter.cpp */
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
#include "com_intel_daal_algorithms_pca_quality_metric_set_QualityMetricSetParameter.h"

using namespace daal::algorithms::pca::quality_metric_set;

/*
* Class:     com_intel_daal_algorithms_pca_quality_metric_set_QualityMetricSetParameter
* Method:    cSetNComponents
* Signature: (JJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_quality_1metric_1set_QualityMetricSetParameter_cSetNComponents
(JNIEnv *, jobject, jlong parAddr, jlong nComponents)
{
    (*(Parameter *)parAddr).nComponents = nComponents;
}

/*
* Class:     com_intel_daal_algorithms_pca_quality_metric_set_QualityMetricSetParameter
* Method:    cGetNComponents
* Signature: (J)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_quality_1metric_1set_QualityMetricSetParameter_cGetNComponents
(JNIEnv *, jobject, jlong parAddr)
{
    return(jlong)(*(Parameter *)parAddr).nComponents;
}

/*
* Class:     com_intel_daal_algorithms_pca_quality_metric_set_QualityMetricSetParameter
* Method:    cSetNBetaReducedModel
* Signature: (JJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_quality_1metric_1set_QualityMetricSetParameter_cSetNFeatures
(JNIEnv *, jobject, jlong parAddr, jlong nFeatures)
{
    (*(Parameter *)parAddr).nFeatures = nFeatures;
}

/*
* Class:     com_intel_daal_algorithms_pca_quality_metric_set_QualityMetricSetParameter
* Method:    cGetNFeatures
* Signature: (J)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_quality_1metric_1set_QualityMetricSetParameter_cGetNFeatures
(JNIEnv *, jobject, jlong parAddr)
{
    return(jlong)(*(Parameter *)parAddr).nFeatures;
}
