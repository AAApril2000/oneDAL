/* file: npy_utils.h */
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
//  Implementation of a heterogeneous table stored as a structure of arrays.
//--
*/

#ifndef __NUMPY_UTILS_TABLE_H__
#define __NUMPY_UTILS_TABLE_H__

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define SET_NPY_FEATURE( _T, _M )                \
    switch(_T) { \
        case NPY_BYTELTR: \
            _M(char);\
            break; \
        case NPY_UBYTELTR:  \
            _M(unsigned char);\
            break; \
        case NPY_SHORTLTR: \
            _M(short);\
            break; \
        case NPY_USHORTLTR: \
            _M(unsigned short); \
            break; \
        case NPY_INTLTR: \
            _M(int); \
            break; \
        case NPY_UINTLTR: \
            _M(unsigned int); \
            break; \
        case NPY_LONGLTR: \
            _M(long); \
            break; \
        case NPY_ULONGLTR: \
            _M(unsigned long); \
            break; \
        case NPY_LONGLONGLTR: \
            _M(long long); \
            break; \
        case NPY_ULONGLONGLTR: \
            _M(unsigned long long); \
            break; \
        case NPY_FLOATLTR: \
        case NPY_CFLOATLTR: \
            _M(float); \
            break; \
        case NPY_DOUBLELTR: \
        case NPY_CDOUBLELTR: \
            _M(double); \
            break; \
        default: \
            std::cerr << "Unsupported NPY type " << (_T) << " ignored\n."; \
            return; \
        };
        // DOUBLELTR FLOATLTR
        // case NPY_STRINGLTR:
        // case NPY_UNICODELTR:
        // case NPY_CLONGDOUBLELTR: break;
        // case NPY_STRINGLTR: break;
        // case NPY_UNICODELTR: break;
        // case NPY_OBJECTLTR: break;
        // case NPY_VOIDLTR: break;
        // case NPY_BOOLLTR: break;
        // case NPY_LONGDOUBLELTR: break;

static inline int get_npy_type(daal::data_management::NumericTableDictionary * dict, size_t index = 0)
{
    switch((*dict)[index].indexType) {
    case daal::data_management::data_feature_utils::DAAL_FLOAT32:
        return NPY_FLOAT32;
    case daal::data_management::data_feature_utils::DAAL_FLOAT64:
        return NPY_FLOAT64;
    case daal::data_management::data_feature_utils::DAAL_INT32_S:
        return NPY_INT32;
    case daal::data_management::data_feature_utils::DAAL_INT32_U:
        return NPY_UINT32;
    case daal::data_management::data_feature_utils::DAAL_INT64_S:
        return NPY_INT64;
    case daal::data_management::data_feature_utils::DAAL_INT64_U:
        return NPY_UINT64;
    case daal::data_management::data_feature_utils::DAAL_INT8_S :
        return NPY_INT8;
    case daal::data_management::data_feature_utils::DAAL_INT8_U :
        return NPY_UINT8;
    case daal::data_management::data_feature_utils::DAAL_INT16_S:
        return NPY_INT16;
    case daal::data_management::data_feature_utils::DAAL_INT16_U:
        return NPY_UINT16;
    default:
        std::cerr << "Unknown IndexType found in DataDictionary. Cannot convert to numpy array.\n";
        return NPY_NOTYPE;
    }
}

#endif // __NUMPY_UTILS_TABLE_H__
