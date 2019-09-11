/* file: train_result.cpp */
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
#include "com_intel_daal_algorithms_decision_tree_classification_training_TrainingResult.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()

namespace dtc = daal::algorithms::decision_tree::classification;
namespace dtct = daal::algorithms::decision_tree::classification::training;

#include "com_intel_daal_algorithms_classifier_training_TrainingResultId.h"
#define modelId  com_intel_daal_algorithms_classifier_training_TrainingResultId_Model

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_training_TrainingResult
* Method:    cGetModel
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_training_TrainingResult_cGetModel
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if (id == modelId)
    {
        return jniArgument<dtct::Result>::get<classifier::training::ResultId, dtc::Model>(resAddr, classifier::training::model);
    }

    return (jlong)0;
}
