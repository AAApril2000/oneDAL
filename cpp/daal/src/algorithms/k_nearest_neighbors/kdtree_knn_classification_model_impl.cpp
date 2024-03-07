/*
 * kNN模型的结构和操作方法，包括模型的创建、序列化（保存）和反序列化（加载）、以及对KD树结构的操作。
*/
// 导入必要的头文件和命名空间
#include "src/algorithms/k_nearest_neighbors/kdtree_knn_classification_model_impl.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

/*
// 定义相关的命名空间和类
*/
namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification // KD树k最近邻分类
{
namespace interface1
{
// 注册序列化类，使模型能够被保存和加载
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_K_NEAREST_NEIGHBOR_MODEL_ID);

// 模型构造函数，初始化特征数量
Model::Model(size_t nFeatures) : daal::algorithms::classifier::Model(), _impl(new ModelImpl(nFeatures)) {}

// 模型析构函数，释放实现部分的内存
Model::~Model()
{
    delete _impl;
}

// 带状态检查的模型构造函数
Model::Model(size_t nFeatures, services::Status & st) : _impl(new ModelImpl(nFeatures))
{
    DAAL_CHECK_COND_ERROR(_impl, st, services::ErrorMemoryAllocationFailed);
}

// 静态方法，用于创建模型实例
services::SharedPtr<Model> Model::create(size_t nFeatures, services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, nFeatures);
}

// 序列化方法，用于保存模型
services::Status Model::serializeImpl(data_management::InputDataArchive * arch)
{
    daal::algorithms::classifier::Model::serialImpl<data_management::InputDataArchive, false>(arch);
    _impl->serialImpl<data_management::InputDataArchive, false>(arch);

    return services::Status();
}

// 反序列化方法，用于加载模型
services::Status Model::deserializeImpl(const data_management::OutputDataArchive * arch)
{
    daal::algorithms::classifier::Model::serialImpl<const data_management::OutputDataArchive, true>(arch);
    _impl->serialImpl<const data_management::OutputDataArchive, true>(arch);

    return services::Status();
}

// 获取模型的特征数量
size_t Model::getNumberOfFeatures() const
{
    return _impl->getNumberOfFeatures();
}

// KD树表的构造函数，初始化存储结构
KDTreeTable::KDTreeTable(size_t rowCount, services::Status & st) : data_management::AOSNumericTable(sizeof(KDTreeNode), 4, rowCount, st)
{
    // 设置KD树节点的属性
    setFeature<size_t>(0, DAAL_STRUCT_MEMBER_OFFSET(KDTreeNode, dimension));
    setFeature<size_t>(1, DAAL_STRUCT_MEMBER_OFFSET(KDTreeNode, leftIndex));
    setFeature<size_t>(2, DAAL_STRUCT_MEMBER_OFFSET(KDTreeNode, rightIndex));
    setFeature<double>(3, DAAL_STRUCT_MEMBER_OFFSET(KDTreeNode, cutPoint));
    st |= allocateDataMemory();
}

// 空的KD树表构造函数
KDTreeTable::KDTreeTable(services::Status & st) : KDTreeTable(0, st) {}

} // namespace interface1

namespace interface3
{
// 参数检查方法，确保参数有效
services::Status Parameter::check() const
{
    DAAL_CHECK_EX(nClasses > 0, services::ErrorIncorrectParameter, services::ParameterName, nClassesStr());
    DAAL_CHECK_EX(k >= 1, services::ErrorIncorrectParameter, services::ParameterName, kStr());
    return services::Status();
}
} // namespace interface3
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
