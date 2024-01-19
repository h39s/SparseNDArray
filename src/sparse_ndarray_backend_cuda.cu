#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/copy.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <unordered_map>

namespace SparseNDArray {
namespace cuda {

namespace py = pybind11;

#define ALIGNMENT 256
typedef uint32_t idx_t;
typedef float scalar_t;

using std::vector;
using std::tuple;
using std::map;
using std::unordered_map;

using thrust::device_vector;
using thrust::host_vector;

struct CudaSparseArray {

    device_vector<idx_t> shape;
    scalar_t default_val;
    device_vector<idx_t> indices;
    device_vector<scalar_t> values;
    size_t size;
    size_t ndim;

    CudaSparseArray(vector<idx_t> shape, scalar_t default_val) {
        this->shape = device_vector<idx_t>(shape);
        this->ndim = shape.size();
        this->default_val = default_val;

        this->indices = device_vector<idx_t>();
        this->values = device_vector<scalar_t>();
        this->size = 0;
    }

    void Fill(scalar_t val) {
        this->default_val = val;
        this->indices.clear();
        this->values.clear();
        this->size = 0;
    }

    vector<idx_t> get_shape() {
        vector<idx_t> out(this->shape.begin(), this->shape.end());
        return out;
    }
};


/***************
 ** UTILITIES **
 ***************/
void FromNumpy(py::array_t<scalar_t> a, CudaSparseArray* out) {
    auto buf = a.request();
    vector<idx_t> shape = vector<idx_t>(buf.shape.begin(), buf.shape.end());
    size_t ndim = shape.size();
    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<idx_t>());

    scalar_t* ptr = (scalar_t*)buf.ptr;

    unordered_map<scalar_t, idx_t> counts;
    out->default_val = ptr[0];
    for (size_t i = 0; i < size; i++) {
        counts[ptr[i]]++;
        if (counts[ptr[i]] > counts[out->default_val]) {
            out->default_val = ptr[i];
        }
    }
    size_t sparse_size = size - counts[out->default_val];
    vector<scalar_t> values = vector<scalar_t>(sparse_size);
    vector<idx_t> indices = vector<idx_t>(sparse_size * ndim);

    size_t sparse_idx = 0;
    for (size_t k = 0; k < size; k++) {
        if (ptr[k] != out->default_val) {
            values[sparse_idx] = ptr[k];
            size_t idx = k;
            for (ssize_t i = ndim-1; i >= 0; i--) {
                indices[sparse_idx*ndim + i] = (idx % shape[i]);
                idx /= shape[i];
            }
            sparse_idx++;
        }
    }
  out->values = device_vector<scalar_t>(values);
  out->indices = device_vector<idx_t>(indices);
  out->size = sparse_size;
  out->ndim = ndim;
}

device_vector<idx_t> compact_strides(device_vector<idx_t> shape) {
  device_vector<idx_t> strides;
  strides.reserve(shape.size());
  idx_t stride = 1;
  for (int i = shape.size() - 1; i >= 0; i--) {
    strides.push_back(stride);
    stride *= shape[i];
  }
  thrust::reverse(strides.begin(), strides.end());
  return strides;
}

py::array_t<scalar_t> ToNumpy(const CudaSparseArray& a) {
    vector<idx_t> shape = vector<idx_t>(a.shape.begin(), a.shape.end());
    size_t ndim = shape.size();
    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<idx_t>());

    py::array_t<scalar_t> out = py::array_t<scalar_t>(shape);
    scalar_t* ptr = (scalar_t*)out.request().ptr;

    for (size_t i = 0; i < size; i++) {
        ptr[i] = a.default_val;
    }

    device_vector<idx_t> strides = compact_strides(a.shape);
    device_vector<idx_t> temp = device_vector<idx_t>(a.ndim);
    for (size_t i = 0; i < a.size; i++) {
        thrust::transform(a.indices.begin() + i*ndim, a.indices.begin() + (i+1)*ndim, strides.begin(), temp.begin(), thrust::multiplies<idx_t>());
        size_t idx = thrust::reduce(temp.begin(), temp.end());
        ptr[idx] = a.values[i];
    }
    return out;
}


/********************
 ** SINGLE ARR OPS **
 ********************/
void EwiseNeg(const CudaSparseArray& a, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = -a.default_val;
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(), thrust::negate<scalar_t>());
    out->size = out->values.size();
}

void EwiseLog(const CudaSparseArray& a, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = std::log(a.default_val);
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(),
                      [] __device__ (scalar_t a) {return std::log(a);} );
    out->size = out->values.size();
}

void EwiseExp(const CudaSparseArray& a, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = std::exp(a.default_val);
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(),
                      [] __device__ (scalar_t a) {return std::exp(a);} );
    out->size = out->values.size();
}

void EwiseTanh(const CudaSparseArray& a, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = std::tanh(a.default_val);
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(),
                      [] __device__ (scalar_t a) {return std::tanh(a);} );
    out->size = out->values.size();
}



/**********************
 ** SCALAR ARITH OPS **
 **********************/
void ScalarAdd(const CudaSparseArray& a, scalar_t val, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = a.default_val + val;
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(), thrust::placeholders::_1 + val);
    out->size = out->values.size();
}

void ScalarSub(const CudaSparseArray& a, scalar_t val, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = a.default_val - val;
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(), thrust::placeholders::_1 - val);
    out->size = out->values.size();
}

void ScalarMul(const CudaSparseArray& a, scalar_t val, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = a.default_val * val;
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(), thrust::placeholders::_1 * val);
    out->size = out->values.size();
}

void ScalarDiv(const CudaSparseArray& a, scalar_t val, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = a.default_val / val;
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(), thrust::placeholders::_1 / val);
    out->size = out->values.size();
}

void ScalarPower(const CudaSparseArray& a, scalar_t val, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = std::pow(a.default_val, val);
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(),
                      [val] __device__ (scalar_t a) {return std::pow(a, val);});
    out->size = out->values.size();
}


/*********************
 ** SCALAR COMP OPS **
 *********************/
void ScalarMaximum(const CudaSparseArray& a, scalar_t val, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = std::max(a.default_val, val);
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(),
                      [val] __device__ (scalar_t a) {return a > val ? a : val;});
    out->size = out->values.size();
}

void ScalarEq(const CudaSparseArray& a, scalar_t val, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = (a.default_val == val) ? 1.0 : 0.0;
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(),
                      [val] __device__ (scalar_t a) {return (a == val) ? 1.0 : 0.0;});
    out->size = out->values.size();
}

void ScalarNe(const CudaSparseArray& a, scalar_t val, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = (a.default_val != val) ? 1.0 : 0.0;
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(),
                      [val] __device__ (scalar_t a) {return (a != val) ? 1.0 : 0.0;});
    out->size = out->values.size();
}

void ScalarGe(const CudaSparseArray& a, scalar_t val, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = (a.default_val >= val) ? 1.0 : 0.0;
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(),
                      [val] __device__ (scalar_t a) {return (a >= val) ? 1.0 : 0.0;});
    out->size = out->values.size();
}

void ScalarLe(const CudaSparseArray& a, scalar_t val, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = (a.default_val <= val) ? 1.0 : 0.0;
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(),
                      [val] __device__ (scalar_t a) {return (a <= val) ? 1.0 : 0.0;});
    out->size = out->values.size();
}

void ScalarGt(const CudaSparseArray& a, scalar_t val, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = (a.default_val > val) ? 1.0 : 0.0;
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(),
                      [val] __device__ (scalar_t a) {return (a > val) ? 1.0 : 0.0;});
    out->size = out->values.size();
}

void ScalarLt(const CudaSparseArray& a, scalar_t val, CudaSparseArray* out) {
    out->indices.resize(a.size * a.ndim);
    thrust::copy(a.indices.begin(), a.indices.end(), out->indices.begin());
    out->default_val = (a.default_val < val) ? 1.0 : 0.0;
    out->values = device_vector<scalar_t>(a.size);
    thrust::transform(a.values.begin(), a.values.end(), out->values.begin(),
                      [val] __device__ (scalar_t a) {return (a < val) ? 1.0 : 0.0;});
    out->size = out->values.size();
}


/**************
 ** EWISE OP **
 **************/
void EwiseOp(const CudaSparseArray a, const CudaSparseArray b, CudaSparseArray* out,
             std::function<scalar_t(scalar_t, scalar_t)> op,
             std::function<size_t(size_t, size_t)> reserve_op,
             bool sparsify=true) {
    if (a.ndim != b.ndim) {
        throw std::invalid_argument("Arrays must have same number of dimensions");
    }
    if (a.shape != b.shape) {
        throw std::invalid_argument("Arrays must have same shape");
    }

    out->shape = a.shape;
    out->ndim = a.ndim;
    out->default_val = op(a.default_val, b.default_val);

    out->values.reserve(reserve_op(a.size, b.size));
    out->indices.reserve(reserve_op(a.size, b.size));

    size_t a_idx = 0;
    size_t b_idx = 0;
    scalar_t new_val;
    while (a_idx < a.size && b_idx < b.size) {
        if (a.indices[a_idx] == b.indices[b_idx]) {
            new_val = op(a.values[a_idx], b.values[b_idx]);
            if (!sparsify || new_val != out->default_val) {
                out->values.push_back(new_val);
                out->indices.push_back(a.indices[a_idx]);
            }
            a_idx++;
            b_idx++;
        } else if (a.indices[a_idx] <= b.indices[b_idx]) {
            new_val = op(a.values[a_idx], b.default_val);
            if (!sparsify || new_val != out->default_val) {
                out->values.push_back(new_val);
                out->indices.push_back(a.indices[a_idx]);
            }
            a_idx++;
        } else if (a.indices[a_idx] >= b.indices[b_idx]) {
            new_val = op(a.default_val, b.values[b_idx]);
            if (!sparsify || new_val != out->default_val) {
                out->values.push_back(new_val);
                out->indices.push_back(b.indices[b_idx]);
            }
            b_idx++;
        } else {
            throw std::runtime_error("Something went wrong");
        }
    }
    while (a_idx < a.size) {
        new_val = op(a.values[a_idx], b.default_val);
        if (!sparsify || new_val != out->default_val) {
            out->values.push_back(new_val);
            out->indices.push_back(a.indices[a_idx]);
        }
        a_idx++;
    }
    while (b_idx < b.size) {
        new_val = op(a.default_val, b.values[b_idx]);
        if (!sparsify || new_val != out->default_val) {
            out->values.push_back(new_val);
            out->indices.push_back(b.indices[b_idx]);
        }
        b_idx++;
    }
    out->size = out->values.size();
}

/*********************
 ** EWISE ARITH OPS **
 *********************/
void EwiseAdd(const CudaSparseArray& a, const CudaSparseArray& b, CudaSparseArray* out) {
    EwiseOp(a, b, out, std::plus<scalar_t>(),
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}

void EwiseSub(const CudaSparseArray& a, const CudaSparseArray& b, CudaSparseArray* out) {
    EwiseOp(a, b, out, std::minus<scalar_t>(),
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}

void EwiseMul(const CudaSparseArray& a, const CudaSparseArray& b, CudaSparseArray* out) {
    EwiseOp(a, b, out, std::multiplies<scalar_t>(),
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}

void EwiseDiv(const CudaSparseArray& a, const CudaSparseArray& b, CudaSparseArray* out) {
    EwiseOp(a, b, out, std::divides<scalar_t>(),
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}


/********************
 ** EWISE COMP OPS **
 ********************/
void EwiseMaximum(const CudaSparseArray& a, const CudaSparseArray& b, CudaSparseArray* out) {
    EwiseOp(a, b, out, [](scalar_t a, scalar_t b) { return std::max(a, b); },
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}

void EwiseEq(const CudaSparseArray& a, const CudaSparseArray& b, CudaSparseArray* out) {
    EwiseOp(a, b, out, [](scalar_t a, scalar_t b) { return (a == b) ? 1 : 0; },
                    [](size_t a_size, size_t b_size) { return std::min(a_size, b_size); });
}

void EwiseNe(const CudaSparseArray& a, const CudaSparseArray& b, CudaSparseArray* out) {
    EwiseOp(a, b, out, [](scalar_t a, scalar_t b) { return (a != b) ? 1 : 0; },
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}

void EwiseGe(const CudaSparseArray& a, const CudaSparseArray& b, CudaSparseArray* out) {
    EwiseOp(a, b, out, [](scalar_t a, scalar_t b) { return (a >= b) ? 1 : 0; },
                    [](size_t a_size, size_t b_size) { return std::min(a_size, b_size); });
}

void EwiseLe(const CudaSparseArray& a, const CudaSparseArray& b, CudaSparseArray* out) {
    EwiseOp(a, b, out, [](scalar_t a, scalar_t b) { return (a <= b) ? 1 : 0; },
                    [](size_t a_size, size_t b_size) { return std::min(a_size, b_size); });
}

void EwiseGt(const CudaSparseArray& a, const CudaSparseArray& b, CudaSparseArray* out) {
    EwiseOp(a, b, out, [](scalar_t a, scalar_t b) { return (a > b) ? 1 : 0; },
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}

void EwiseLt(const CudaSparseArray& a, const CudaSparseArray& b, CudaSparseArray* out) {
    EwiseOp(a, b, out, [](scalar_t a, scalar_t b) { return (a < b) ? 1 : 0; },
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}

} // namespace cuda
} // namespace SparseNDArray


/*****************
 ** PYBIND CODE **
 *****************/
PYBIND11_MODULE(sparse_ndarray_backend_cuda, m) {
    using namespace SparseNDArray;
    using namespace cuda;

    m.attr("__device_name__") = "cuda";

    py::class_<CudaSparseArray>(m, "SparseArray")
        .def(py::init<vector<idx_t>, scalar_t>(), py::return_value_policy::take_ownership)
        .def("fill", &CudaSparseArray::Fill)
        .def("get_shape", &CudaSparseArray::get_shape)
        .def_readonly("size", &CudaSparseArray::size)
        .def_readonly("ndim", &CudaSparseArray::ndim)
        .def_readwrite("default_val", &CudaSparseArray::default_val);

    // UTILITIES
    m.def("from_numpy", FromNumpy);
    m.def("to_numpy", ToNumpy);

    // SINGLE ARR OPS
    m.def("ewise_neg", EwiseNeg);
    m.def("ewise_log", EwiseLog);
    m.def("ewise_exp", EwiseExp);
    m.def("ewise_tanh", EwiseTanh);

    // SCALAR ARITH OPS
    m.def("scalar_add", ScalarAdd);
    m.def("scalar_sub", ScalarSub);
    m.def("scalar_mul", ScalarMul);
    m.def("scalar_div", ScalarDiv);
    m.def("scalar_power", ScalarPower);

    // SCALAR COMP OPS
    m.def("scalar_maximum", ScalarMaximum);
    m.def("scalar_eq", ScalarEq);
    m.def("scalar_ne", ScalarNe);
    m.def("scalar_ge", ScalarGe);
    m.def("scalar_le", ScalarLe);
    m.def("scalar_gt", ScalarGt);
    m.def("scalar_lt", ScalarLt);

    // EWISE ARITH OPS
    m.def("ewise_add", EwiseAdd);
    m.def("ewise_sub", EwiseSub);
    m.def("ewise_mul", EwiseMul);
    m.def("ewise_div", EwiseDiv);

    // EWISE COMP OPS
    m.def("ewise_maximum", EwiseMaximum);
    m.def("ewise_eq", EwiseEq);
    m.def("ewise_ne", EwiseNe);
    m.def("ewise_ge", EwiseGe);
    m.def("ewise_le", EwiseLe);
    m.def("ewise_gt", EwiseGt);
    m.def("ewise_lt", EwiseLt);

    // // MATMUL OP
    // m.def("matmul", matmul);

    // // ARRAY MANIP OPS
    // m.def("reshape", reshape);
    // m.def("permute", permute);
    // m.def("broadcast_to", broadcast_to);

    // // REDUCTION OPS
    // m.def("reduce_max", reduce_max);
    // m.def("reduce_sum", reduce_sum);

    // // GETITEM SETITEM OPS
    // m.def("getitem", getitem);
    // m.def("ewise_setitem", EwiseSetitem);
    // m.def("scalar_setitem", ScalarSetitem);
}
