/* 训练bf_KNN分类算法的容器类
 * 1.初始化容器：这个类根据运行环境（CPU或非CPU设备）初始化不同的计算内核。
 * 如果是CPU环境，它使用`__DAAL_INITIALIZE_KERNELS`宏初始化标准内核；
 * 对于非CPU环境（比如使用SYCL的GPU），它使用`__DAAL_INITIALIZE_KERNELS_SYCL`宏初始化专门的UCAPI内核。
 * 2.实现了`compute`方法。函数内部，准备数据和标签，然后根据环境调用适当的内核执行训练过程。
 * 数据处理：在`compute`方法内，算法会根据用户的参数决定是否将数据复制到模型中，以及是否需要计算类标签。
 */

// 包含必要的头文件
#include "services/internal/sycl/execution_context.h" // 引入SYCL执行上下文相关功能
#include "src/algorithms/kernel.h" // 引入算法核心功能
#include "data_management/data/numeric_table.h" // 用于数据管理的数值表
#include "services/daal_shared_ptr.h" // 智能指针支持
#include "algorithms/classifier/classifier_model.h" // 分类器模型基类
#include "algorithms/k_nearest_neighbors/bf_knn_classification_training_batch.h" // bf_KNN训练批处理类
#include "src/algorithms/k_nearest_neighbors/oneapi/bf_knn_classification_train_kernel_ucapi.h" // 针对UCAPI的KNN训练内核
#include "src/algorithms/k_nearest_neighbors/oneapi/bf_knn_classification_model_ucapi_impl.h" // UCAPI下的KNN模型实现
#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_train_kernel.h" // KNN训练内核
//通过UCAPI(通用计算API)，可以编写一次代码，然后在多种硬件平台上运行，无需针对每种设备编写特定的代码。

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace training
{
using namespace daal::data_management; // 使用数据管理命名空间简化代码

template <typename algorithmFpType, training::Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    // 根据运行环境选择初始化CPU或GPU相关的内核
    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    // 如果是CPU环境，初始化CPU相关的内核
    if (deviceInfo.isCpu)
    {
        __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationTrainKernel, algorithmFpType);
    }
    else // 如果是GPU环境，初始化GPU相关的内核
    {
        __DAAL_INITIALIZE_KERNELS_SYCL(internal::KNNClassificationTrainKernelUCAPI, DAAL_FPTYPE);
    }
}

template <typename algorithmFpType, training::Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::~BatchContainer()
{
    // 销毁内核资源
    __DAAL_DEINITIALIZE_KERNELS();
}

/*********************************************************************************************************************/
/* compute函数，执行训练过程
 * 准备训练所需的数据和标签。
 * 根据参数设置决定数据和标签是否复制到模型中。
 * 根据设备类型（CPU或其他）选择合适的计算核心执行训练。
 * 记录并检查整个执行过程中的状态，确保训练过程顺利进行。
 */
template <typename algorithmFpType, training::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFpType, method, cpu>::compute()
{
    // 状态对象，用于记录执行过程中的状态，eg：是否成功执行算法
    services::Status status;

    // 获取算法参数(_par)、输入(_in)和输出(_res)对象，并将它们转换为更具体的类型
    // 这里的par、input、result分别表示算法的参数、输入和结果对象
    const bf_knn_classification::Parameter * const par         = static_cast<bf_knn_classification::Parameter *>(_par);
    const bf_knn_classification::training::Input * const input = static_cast<bf_knn_classification::training::Input *>(_in);
    Result * const result                                      = static_cast<Result *>(_res);

    // 从输入对象中获取训练数据
    const NumericTablePtr x = input->get(classifier::training::data);

    // 从结果对象中获取模型
    const bf_knn_classification::ModelPtr r = result->get(classifier::training::model);

    // 获取DAAL环境的引用
    daal::services::Environment::env & env = *_env;

    // 根据参数决定是否在模型中使用输入数据
    const bool copy = (par->dataUseInModel == doNotUse);
    // 设置模型的数据，如果copy为true，则复制数据到模型中，否则直接使用
    status |= r->impl()->setData<algorithmFpType>(x, copy);

    // 如果需要计算类标签，则从输入对象中获取标签并设置到模型中
    if ((par->resultsToEvaluate & daal::algorithms::classifier::computeClassLabels) != 0)
    {
        const NumericTablePtr y = input->get(classifier::training::labels);
        status |= r->impl()->setLabels<algorithmFpType>(y, copy);
    }
    // 检查到目前为止是否有错误发生
    DAAL_CHECK_STATUS_VAR(status);

    // 获取默认上下文和设备信息，以决定是使用CPU还是其他设备执行算法
    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    // 如果设备是CPU，则调用CPU版本的核心算法
    if (deviceInfo.isCpu)
    {
        __DAAL_CALL_KERNEL(env, internal::KNNClassificationTrainKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFpType), compute, 
                           r->impl()->getData().get(), r->impl()->getLabels().get(), r.get(), *par, *par->engine);
    }
    // 如果设备不是CPU，则调用支持SYCL的核心算法
    else
    {
        __DAAL_CALL_KERNEL_SYCL(env, internal::KNNClassificationTrainKernelUCAPI, __DAAL_KERNEL_ARGUMENTS(algorithmFpType), compute,
                                r->impl()->getData().get(), r->impl()->getLabels().get(), r.get(), *par, *par->engine);
    }
}


} // namespace training
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
