/* file: kdtree_knn_classification_predict_dense_default_batch_fpt_dispatcher.cpp */
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

/*
//++
//  Implementation of K-Nearest Neighbors algorithm container - a class that contains fast K-Nearest Neighbors prediction kernels for supported
//  architectures.
//--
*/

#include "kdtree_knn_classification_predict_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(kdtree_knn_classification::prediction::BatchContainer, batch, DAAL_FPTYPE, kdtree_knn_classification::prediction::defaultDense)
} // namespace algorithms
} // namespace daal
