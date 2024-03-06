// 引入k最近邻分类算法的训练类型相关的头文件
#include "algorithms/k_nearest_neighbors/bf_knn_classification_training_types.h"
// 引入序列化工具相关的头文件
#include "src/services/serialization_utils.h"

// 使用命名空间简化代码
using namespace daal::data_management;
using namespace daal::services;

// 定义一个名为daal的命名空间
namespace daal
{
// 在daal命名空间中定义algorithms命名空间
namespace algorithms
{
// 在algorithms命名空间中定义bf_knn_classification命名空间，代表基于暴力方法的K最近邻分类
namespace bf_knn_classification
{
// 在bf_knn_classification命名空间中定义training命名空间，用于处理训练过程
namespace training
{
// 在training命名空间中定义interface1命名空间，作为接口使用
namespace interface1
{
// 注册序列化类，用于在训练结果的序列化和反序列化过程中使用
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_K_NEAREST_NEIGHBOR_BF_TRAINING_RESULT_ID);

// 定义Result类的构造函数，继承自classifier::training::Result类
Result::Result() : classifier::training::Result() {}

// 定义Result类的get方法，用于获取训练结果
// 方法接受一个classifier::training::ResultId枚举类型的id参数，用于指定需要获取的训练结果的类型
// 方法返回一个指向bf_knn_classification::Model的智能指针，这个模型包含了训练的结果
daal::algorithms::bf_knn_classification::ModelPtr Result::get(classifier::training::ResultId id) const
{
    // 使用staticPointerCast方法将基类SerializationIface的指针转换为bf_knn_classification::Model的指针
    // 这是一个类型安全的转换，保证了转换的正确性
    return services::staticPointerCast<daal::algorithms::bf_knn_classification::Model, data_management::SerializationIface>(Argument::get(id));
}

} // 结束interface1命名空间
} // 结束training命名空间
} // 结束bf_knn_classification命名空间
} // 结束algorithms命名空间
} // 结束daal命名空间
