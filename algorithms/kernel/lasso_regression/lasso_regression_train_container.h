/* file: lasso_regression_train_container.h */
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
//  Implementation of lasso regression container.
//--
*/

#ifndef __LASSO_REGRESSION_TRAIN_CONTAINER_H__
#define __LASSO_REGRESSION_TRAIN_CONTAINER_H__

#include "kernel.h"
#include "algorithms/lasso_regression/lasso_regression_training_types.h"
#include "algorithms/lasso_regression/lasso_regression_training_batch.h"
#include "lasso_regression_train_kernel.h"
#include "lasso_regression_model_impl.h"
#include "algorithms/optimization_solver/sgd/sgd_batch.h"
#include "service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace lasso_regression
{
namespace training
{

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::TrainBatchKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);
    auto x = input->get(data);
    auto y = input->get(dependentVariables);
    NumericTablePtr gramMatrix(input->get(training::gramMatrix));
    lasso_regression::Model *m = result->get(model).get();
    const lasso_regression::training::Parameter *par = static_cast<lasso_regression::training::Parameter*>(_par);
    daal::services::Environment::env &env = *_env;
    services::SharedPtr<daal::algorithms::optimization_solver::mse::Batch<algorithmFPType> > objFunc(new daal::algorithms::optimization_solver::mse::Batch<algorithmFPType>(x->getNumberOfRows()));
    __DAAL_CALL_KERNEL(env, internal::TrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
        compute, daal::services::internal::getHostApp(*input), x, y, *m, *result, *par, objFunc);
}

}
}
}
}
#endif
