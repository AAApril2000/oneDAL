/* file: softmax_layer_forward_types.h */
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
//  Implementation of the forward softmax layer interface.
//--
*/

#ifndef __SOFTMAX_LAYER_FORWARD_TYPES_H__
#define __SOFTMAX_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"
#include "algorithms/neural_networks/layers/softmax/softmax_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace softmax
{
/**
 * @defgroup softmax_layers_forward Forward Softmax Layer
 * \copydoc daal::algorithms::neural_networks::layers::softmax::forward
 * @ingroup softmax_layers
 * @{
 */
/**
 * \brief Contains classes of the forward softmax layer
 */
namespace forward
{
/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward softmax layer
 */
class DAAL_EXPORT Input : public layers::forward::Input
{
public:
    typedef layers::forward::Input super;
    /** Default constructor */
    Input();

    /** Copy constructor */
    Input(const Input& other);

    /**
     * Returns an input object for the forward softmax layer
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    using layers::forward::Input::get;

    /**
     * Sets an input object for the forward softmax layer
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    using layers::forward::Input::set;

    /**
     * Returns dimensions of weights tensor
     * \return Dimensions of weights tensor
     */
    virtual const services::Collection<size_t> getWeightsSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE;

    /**
     * Returns dimensions of biases tensor
     * \return Dimensions of biases tensor
     */
    virtual const services::Collection<size_t> getBiasesSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE;

    virtual ~Input() {}

    /**
     * Checks input object of the forward softmax layer
     * \param[in] par     Layer parameter
     * \param[in] method  Computation method of the layer
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the forward softmax layer
 */
class DAAL_EXPORT Result : public layers::forward::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    /** Default constructor */
    Result();
    virtual ~Result() {};

    /**
     * Returns result of the forward softmax layer
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    using layers::forward::Result::get;

    /**
     * Sets the result of the forward softmax layer
     * \param[in] id      Identifier of the result
     * \param[in] value   Pointer to the object
     */
    using layers::forward::Result::set;

    /**
     * Returns result of the forward softmax layer
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    data_management::TensorPtr get(LayerDataId id) const;

    /**
     * Sets the result of the forward softmax layer
     * \param[in] id      Identifier of the result
     * \param[in] value   Pointer to the object
     */
    void set(LayerDataId id, const data_management::TensorPtr &value);

    /**
     * Checks the result of the forward softmax layer
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns dimensions of value tensor
     * \return Dimensions of value tensor
     */
    virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                            const daal::algorithms::Parameter *par, const int method) const DAAL_C11_OVERRIDE;

    /**
    * Allocates memory to store the result of the forward softmax layer
    * \param[in] input     Pointer to an object containing the input data
    * \param[in] par      %Parameter of the backward softmax layer
    * \param[in] method    Computation method
    *
     * \return Status of computations
    */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

    /**
     * Sets the result that is used in backward softmax layer
     * \param[in] input     Pointer to an object containing the input data
     *
     * \return Status of computations
     */
    virtual services::Status setResultForBackward(const daal::algorithms::Input *input) DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

} // namespace interface1
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
} // namespace forward
/** @} */
} // namespace softmax
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
