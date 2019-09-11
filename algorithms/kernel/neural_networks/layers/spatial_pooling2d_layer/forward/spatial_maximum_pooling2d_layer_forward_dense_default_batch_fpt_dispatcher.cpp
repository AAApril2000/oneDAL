/* file: spatial_maximum_pooling2d_layer_forward_dense_default_batch_fpt_dispatcher.cpp */
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

//++
//  Implementation of forward pooling layer container.
//--


#include "spatial_maximum_pooling2d_layer_forward_batch_container.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace forward
{
__DAAL_INSTANTIATE_DISPATCH_LAYER_CONTAINER_FORWARD(neural_networks::layers::spatial_maximum_pooling2d::forward::interface1::BatchContainer, DAAL_FPTYPE,
                                      neural_networks::layers::spatial_maximum_pooling2d::defaultDense)
}
}
}
}
}
