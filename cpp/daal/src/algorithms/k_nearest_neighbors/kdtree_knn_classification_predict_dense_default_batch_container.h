/*
// 本文件实现了K-最近邻算法的容器类。这个类包含了为支持的架构优化的快速K-最近邻预测核心。
*/

// 包含必要的头文件
#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_predict.h" // KNN分类预测的主要声明
#include "src/algorithms/k_nearest_neighbors/kdtree_knn_classification_predict_dense_default_batch.h" // KNN预测的具体实现

// 使用daal命名空间简化代码
namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification // KD树k最近邻分类命名空间
{
namespace prediction // 预测子命名空间
{
namespace interface3 // 版本3的接口
{
// 模板类BatchContainer的定义，支持不同的浮点类型、方法和CPU类型
template <typename algorithmFpType, Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : PredictionContainerIface()
{
    // 根据算法的浮点类型和方法，初始化KNN分类预测的内核
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationPredictKernel, algorithmFpType, method);
}

// 析构函数，用于释放初始化的内核资源
template <typename algorithmFpType, Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

// compute函数，执行预测计算
template <typename algorithmFpType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFpType, method, cpu>::compute()
{
    // 类型转换输入和结果对象，以便使用
    const classifier::prediction::Input * const input = static_cast<const classifier::prediction::Input *>(_in);
    Result * const result                             = static_cast<Result *>(_res);

    // 从输入中获取数据、模型以及结果的预测、索引和距离
    const data_management::NumericTableConstPtr a = input->get(classifier::prediction::data);
    const classifier::ModelConstPtr m             = input->get(classifier::prediction::model);
    const data_management::NumericTablePtr r      = result->get(prediction::prediction);

    // 获取算法参数
    const Parameter * const par = static_cast<const Parameter *>(_par);

    // 根据参数设置，准备索引和距离的输出
    data_management::NumericTablePtr indices;
    data_management::NumericTablePtr distances;
    if (par->resultsToCompute & computeIndicesOfNeighbors)
    {
        indices = result->get(prediction::indices);
    }
    if (par->resultsToCompute & computeDistances)
    {
        distances = result->get(prediction::distances);
    }

    // 获取当前环境
    daal::services::Environment::env & env = *_env;

    // 调用内核的compute函数，执行预测
    __DAAL_CALL_KERNEL(env, internal::KNNClassificationPredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFpType, method), compute, a.get(), m.get(),
                       r.get(), indices.get(), distances.get(), par);
}

} // namespace interface3
} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
