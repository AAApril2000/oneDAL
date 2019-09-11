/* file: pca_dense_svd_batch_v1_fpt_dispatcher.cpp */
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
//  Implementation of PCA SVD algorithm container.
//--
*/

#include "pca/inner/pca_batch_v1.h"
#include "pca/inner/pca_dense_svd_batch_container_v1.h"
#include "pca_dense_svd_batch_kernel.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER_KM(pca::interface1::BatchContainer, batch, DAAL_FPTYPE, pca::svdDense)
}
} // namespace daal
