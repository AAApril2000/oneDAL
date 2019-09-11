/* file: model_builder.cpp */
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
#include "com_intel_daal_algorithms_gbt_classification_ModelBuilder.h"
#include "common_helpers_functions.h"

using namespace daal;
using namespace daal::algorithms::gbt::classification;
using namespace daal::data_management;
using namespace daal::services;

/*
* Class:     com_intel_daal_algorithms_gbt_classification_ModelBuilder
* Method:    cInit
* Signature: (JIII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_classification_ModelBuilder_cInit
(JNIEnv *env, jobject, jlong nFeatures, jlong nIterations, jlong nClasses)
{
    jlong modelBuilderAddr = (jlong)(new SharedPtr<ModelBuilder>(new ModelBuilder(nFeatures, nIterations, nClasses)));

    services::SharedPtr<ModelBuilder> *ptr = new services::SharedPtr<ModelBuilder>();
    *ptr = staticPointerCast<ModelBuilder>(*(SharedPtr<ModelBuilder> *)modelBuilderAddr);
    DAAL_CHECK_THROW((*ptr)->getStatus());

    return modelBuilderAddr;
}

/*
* Class:     com_intel_daal_algorithms_gbt_classification_ModelBuilder
* Method:    cCreateTree
* Signature: (JIII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_classification_ModelBuilder_cCreateTree
(JNIEnv *env, jobject, jlong algAddr, jlong nNodes, jlong classLabel)
{
    services::SharedPtr<ModelBuilder> *ptr = new services::SharedPtr<ModelBuilder>();
    *ptr = staticPointerCast<ModelBuilder>(*(SharedPtr<ModelBuilder> *)algAddr);
    long treeId = (*ptr)->createTree(nNodes, classLabel);
    DAAL_CHECK_THROW((*ptr)->getStatus());
    return treeId;
}

/*
* Class:     com_intel_daal_algorithms_gbt_classification_ModelBuilder
* Method:    cAddSplitNode
* Signature: (JIII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_classification_ModelBuilder_cAddSplitNode
(JNIEnv *env, jobject, jlong algAddr, jlong treeId, jlong parentId, jlong position, jlong featureIndex, jdouble featureValue)
{
    services::SharedPtr<ModelBuilder> *ptr = new services::SharedPtr<ModelBuilder>();
    *ptr = staticPointerCast<ModelBuilder>(*(SharedPtr<ModelBuilder> *)algAddr);
    long nodeId = (*ptr)->addSplitNode(treeId, parentId, position, featureIndex, static_cast<double>(featureValue));
    DAAL_CHECK_THROW((*ptr)->getStatus());
    return nodeId;
}

/*
* Class:     com_intel_daal_algorithms_gbt_classification_ModelBuilder
* Method:    cAddLeafNode
* Signature: (JIII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_classification_ModelBuilder_cAddLeafNode
(JNIEnv *env, jobject, jlong algAddr, jlong treeId, jlong parentId, jlong position, jdouble response)
{
    services::SharedPtr<ModelBuilder> *ptr = new services::SharedPtr<ModelBuilder>();
    *ptr = staticPointerCast<ModelBuilder>(*(SharedPtr<ModelBuilder> *)algAddr);
    long nodeId = (*ptr)->addLeafNode(treeId, parentId, position, response);
    DAAL_CHECK_THROW((*ptr)->getStatus());
    return nodeId;
}

/*
 * Class:     com_intel_daal_algorithms_gbt_classification_ModelBuilder
 * Method:    cGetModel
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_classification_ModelBuilder_cGetModel
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    services::SharedPtr<ModelBuilder> *ptr = new services::SharedPtr<ModelBuilder>();
    *ptr = staticPointerCast<ModelBuilder>(*(SharedPtr<ModelBuilder> *)algAddr);
    ModelPtr *model = new ModelPtr;
    *model = staticPointerCast<Model>((*ptr)->getModel());
    return (jlong)model;
}
