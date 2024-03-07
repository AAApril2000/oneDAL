/*
KNN分类预测的容器实现。这个容器负责处理算法参数和输入输出数据的准备工作，初始化计算资源，执行预测计算，并在完成后释放资源。
根据运行环境（CPU或其他设备）选择不同的内核实现，以优化性能。

*/
/* 包含必要的头文件 */
#include "algorithms/k_nearest_neighbors/bf_knn_classification_predict.h" // KNN分类预测的基础定义
#include "src/algorithms/k_nearest_neighbors/oneapi/bf_knn_classification_predict_kernel_ucapi.h" // 使用统一通用编程接口（UCAPI）的KNN预测内核定义
#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_predict_kernel.h" // KNN预测内核的基本实现
#include "src/algorithms/k_nearest_neighbors/oneapi/bf_knn_classification_model_ucapi_impl.h" // KNN模型UCAPI实现
#include "services/error_indexes.h" // 错误索引服务

/* 定义命名空间 */
namespace daal
{
namespace algorithms
{
namespace bf_knn_classification // 暴力法KNN分类
{
namespace prediction // 预测命名空间
{
/* BatchContainer模板定义 */
template <typename algorithmFpType, Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
: PredictionContainerIface() // 初始化预测容器接口
{
    auto & context    = services::internal::getDefaultContext(); // 获取默认上下文
    auto & deviceInfo = context.getInfoDevice(); // 获取设备信息

    if (deviceInfo.isCpu) // 如果是在CPU上执行
    {
        // 初始化CPU上的KNN预测内核
        __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationPredictKernel, algorithmFpType);
    }
    else // 如果在其他设备上执行
    {
        // 初始化使用UCAPI的KNN预测内核
        __DAAL_INITIALIZE_KERNELS_SYCL(internal::KNNClassificationPredictKernelUCAPI, algorithmFpType);
    }
}

/* 析构函数 */
template <typename algorithmFpType, Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS(); // 释放内核资源
}

/* 计算函数 */
template <typename algorithmFpType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFpType, method, cpu>::compute()
{
    // 类型转换输入和结果
    const classifier::prediction::Input * const input        = static_cast<const classifier::prediction::Input *>(_in);
    bf_knn_classification::prediction::Result * const result = static_cast<bf_knn_classification::prediction::Result *>(_res);
    
    // 获取输入数据
    const data_management::NumericTableConstPtr a            = input->get(classifier::prediction::data);
    const classifier::ModelConstPtr m                        = input->get(classifier::prediction::model);
    
    // 准备输出数据
    const data_management::NumericTablePtr label             = result->get(bf_knn_classification::prediction::prediction);
    const data_management::NumericTablePtr indices           = result->get(bf_knn_classification::prediction::indices);
    const data_management::NumericTablePtr distances         = result->get(bf_knn_classification::prediction::distances);
    
    auto & context                                           = services::internal::getDefaultContext();
    auto & deviceInfo                                        = context.getInfoDevice();
    
    const Parameter * const par                              = static_cast<const Parameter *>(_par); // 获取算法参数

    internal::KernelParameter kernelPar; // 内部核心参数
    
    // 设置内部核心参数
    kernelPar.nClasses          = par->nClasses;
    kernelPar.k                 = par->k;
    kernelPar.dataUseInModel    = par->dataUseInModel;
    kernelPar.resultsToCompute  = par->resultsToCompute;
    kernelPar.voteWeights       = par->voteWeights;
    kernelPar.engine            = par->engine->clone();
    kernelPar.resultsToEvaluate = par->resultsToEvaluate;

    if (deviceInfo.isCpu) // 如果在CPU上执行
    {
        // 调用CPU内核计算函数
        __DAAL_CALL_KERNEL(env, internal::KNNClassificationPredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFpType), compute, a.get(), m.get(),
                           label.get(), indices.get(), distances.get(), &kernelPar);
    }
    else // 如果在其他设备上执行
    {
        // 调用使用UCAPI的内核计算函数
        __DAAL_CALL_KERNEL_SYCL(env, internal::KNNClassificationPredictKernelUCAPI, __DAAL_KERNEL_ARGUMENTS(algorithmFpType), compute, a.get(),
                                m.get(), label.get(), indices.get(), distances.get(), par);
    }
}

} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal
