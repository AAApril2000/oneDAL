/* 
 * Brute-Force, BF分类模型的实现。
 * 模型的定义、序列化（保存模型以便之后使用）和反序列化（加载模型进行预测或其他操作），以及参数验证。

 */
// 引入相关的头文件
#include "src/algorithms/k_nearest_neighbors/oneapi/bf_knn_classification_model_ucapi_impl.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"
#include "src/services/service_data_utils.h"

// 使用Intel DAAL库的命名空间，简化代码
using namespace daal::data_management;
using namespace daal::services;

// 定义代码在DAAL的命名空间中
namespace daal
{
namespace algorithms // 算法相关的命名空间
{
namespace bf_knn_classification // 特定于暴力搜索k-NN分类的命名空间
{
namespace interface1 // 版本1的接口，提供向后兼容性
{
// 为模型注册序列化类，允许模型数据的保存和加载
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_K_NEAREST_NEIGHBOR_BF_MODEL_ID);

// 构造函数，用于创建一个新的Model对象，其中包含nFeatures特征数量
Model::Model(size_t nFeatures) : daal::algorithms::classifier::Model(), _impl(new ModelImpl(nFeatures)) {}

// 析构函数，用于清理资源，主要是删除实现的实例
Model::~Model()
{
    delete _impl;
    _impl = nullptr;
}

// 另一个构造函数，这次包括了一个状态参数，以检查对象的创建是否成功
Model::Model(size_t nFeatures, services::Status & st) : _impl(new ModelImpl(nFeatures))
{
    DAAL_CHECK_COND_ERROR(_impl, st, services::ErrorMemoryAllocationFailed);
}

// 序列化函数，用于将模型数据保存到archive中
services::Status Model::serializeImpl(data_management::InputDataArchive * arch)
{
    // 调用基类的序列化函数
    daal::algorithms::classifier::Model::serialImpl<data_management::InputDataArchive, false>(arch);
    // 调用_impl成员的序列化实现
    return _impl->serialImpl<data_management::InputDataArchive, false>(arch);
}

// 反序列化函数，用于从archive中加载模型数据
services::Status Model::deserializeImpl(const data_management::OutputDataArchive * arch)
{
    // 调用基类的反序列化函数
    daal::algorithms::classifier::Model::serialImpl<const data_management::OutputDataArchive, true>(arch);
    // 调用_impl成员的反序列化实现
    return _impl->serialImpl<const data_management::OutputDataArchive, true>(arch);
}

// 获取模型的特征数量
size_t Model::getNumberOfFeatures() const
{
    return _impl->getNumberOfFeatures();
}

// 检查参数设置是否正确，比如类别数nClasses和近邻数k
services::Status Parameter::check() const
{
    // 检查类别数，它应该大于1且小于最大整数值
    DAAL_CHECK_EX(this->nClasses > 1 && this->nClasses < static_cast<size_t>(services::internal::MaxVal<int>::get()),
                  services::ErrorIncorrectParameter, services::ParameterName, nClassesStr());
    // 检查近邻数k，它应该大于0且小于最大整数值
    DAAL_CHECK_EX(this->k > 0 && this->k <= static_cast<size_t>(services::internal::MaxVal<int>::get()), services::ErrorIncorrectParameter,
                  services::ParameterName, kStr());
    return services::Status();
}

} // namespace interface1
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal
