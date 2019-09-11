/* file: ridge_regression_train_dense_normeq_impl.i */
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
//  Implementation of auxiliary functions for ridge regression Normal Equations (normEqDense) method.
//--
*/

#ifndef __RIDGE_REGRESSION_TRAIN_DENSE_NORMEQ_IMPL_I__
#define __RIDGE_REGRESSION_TRAIN_DENSE_NORMEQ_IMPL_I__

#include "ridge_regression_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
namespace internal
{
using namespace daal::algorithms::linear_model::normal_equations::training::internal;

template <typename algorithmFPType, CpuType cpu>
Status BatchKernel<algorithmFPType, training::normEqDense, cpu>::compute(const NumericTable &x,
                                                                         const NumericTable &y,
                                                                         NumericTable &xtx,
                                                                         NumericTable &xty,
                                                                         NumericTable &beta,
                                                                         bool interceptFlag,
                                                                         const NumericTable &ridge) const
{
    Status st = UpdateKernelType::compute(x, y, xtx, xty, true, interceptFlag);
    if (st)
        st = FinalizeKernelType::compute(xtx, xty, xtx, xty, beta, interceptFlag,
                                         KernelHelper<algorithmFPType, cpu>(ridge));
    return st;
}

template <typename algorithmFPType, CpuType cpu>
Status OnlineKernel<algorithmFPType, training::normEqDense, cpu>::compute(
    const NumericTable &x, const NumericTable &y, NumericTable &xtx, NumericTable &xty,
    bool interceptFlag) const
{
    return UpdateKernelType::compute(x, y, xtx, xty, false, interceptFlag);
}

template <typename algorithmFPType, CpuType cpu>
Status OnlineKernel<algorithmFPType, training::normEqDense, cpu>::finalizeCompute(
    const NumericTable &xtx, const NumericTable &xty, NumericTable &xtxFinal, NumericTable &xtyFinal,
    NumericTable &beta, bool interceptFlag, const NumericTable &ridge) const
{
    return FinalizeKernelType::compute(xtx, xty, xtxFinal, xtyFinal, beta, interceptFlag,
                                      KernelHelper<algorithmFPType, cpu>(ridge));
}

} // namespace internal
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
