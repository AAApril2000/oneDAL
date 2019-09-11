/* file: pca_dense_svd_batch_container_v2.h */
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
//  Implementation of PCA Correlation algorithm container.
//--
*/

#ifndef __PCA_DENSE_SVD_BATCH_CONTAINER_V2_H__
#define __PCA_DENSE_SVD_BATCH_CONTAINER_V2_H__

#include "pca/inner/pca_batch_v2.h"
#include "kernel.h"
#include "pca_dense_svd_batch_kernel.h"
#include "pca_dense_svd_container.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface2
{

template <typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, svdDense, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PCASVDBatchKernel, algorithmFPType, interface2::BatchParameter
                       <algorithmFPType, pca::svdDense>);
}

template <typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, svdDense, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, CpuType cpu>
Status BatchContainer<algorithmFPType, svdDense, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);
    interface2::BatchParameter<algorithmFPType, pca::svdDense>* parameter = static_cast<interface2::BatchParameter<algorithmFPType,
                                                                                        pca::svdDense> *>(_par);

    internal::InputDataType dtype = getInputDataType(input);

    data_management::NumericTablePtr data = input->get(pca::data);
    data_management::NumericTablePtr eigenvalues  = result->get(pca::eigenvalues);
    data_management::NumericTablePtr eigenvectors = result->get(pca::eigenvectors);
    data_management::NumericTablePtr means        = result->get(pca::means);
    data_management::NumericTablePtr variances    = result->get(pca::variances);

    auto normalizationAlgorithm = parameter->normalization;
    normalizationAlgorithm->input.set(normalization::zscore::data, data);
    auto algParameter = normalizationAlgorithm->getParameter();
    if (parameter->resultsToCompute & mean)
    {
        algParameter->resultsToCompute |= normalization::zscore::mean;
    }

    if (parameter->resultsToCompute & variance)
    {
        algParameter->resultsToCompute |= normalization::zscore::variance;
    }

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PCASVDBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, interface2::BatchParameter
                       <algorithmFPType, pca::svdDense>), compute, dtype, *data, parameter, *eigenvalues, *eigenvectors, *means, *variances);
}

} // namespace interface2
} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
