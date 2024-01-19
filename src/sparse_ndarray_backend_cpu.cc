#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <unordered_map>
#include <map>
#include <deque>
#include <cstdint>
#include <algorithm>
#include <limits>


namespace SparseNDArray {
namespace cpu {

#define ALIGNMENT 256
typedef float scalar_t;

using std::vector;
using std::tuple;
using std::unordered_map;
using std::map;
namespace py = pybind11;

typedef uint32_t idx_t;

/******************
 ** SPARSE ARRAY **
 ******************/
struct SparseArray {
    /* Sparse array class for CPU backend.
     *
     * Attributes:
     *   size: Number of non-default elements.
     *   ndim: Number of dimensions.
     *   shape: Shape of the array.
     *   indices: Indices of non-default elements.
     *            Each index is a vector of length ndim.
     *   values: Values of non-default elements.
     *   default_val: Default value of the array.
    */
    vector<idx_t> shape;
    scalar_t default_val;
    vector<vector<idx_t>> indices;
    vector<scalar_t> values;
    size_t size;
    size_t ndim;

    // scalar_t* ptr;
    // size_t capacity;

    SparseArray(vector<idx_t> shape, scalar_t default_val=0) {
        this->shape = shape;
        this->ndim = shape.size();
        this->default_val = default_val;

        this->indices = vector<vector<idx_t>>();
        this->values = vector<scalar_t>();
        this->size = 0;

        // this->capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<idx_t>());
        // int ret = posix_memalign((void**)&this->ptr, ALIGNMENT, this->capacity * ELEM_SIZE);
        // if (ret != 0) throw std::bad_alloc();
    }
    ~SparseArray() {}

    void Fill(scalar_t val) {
        this->default_val = val;
        this->indices = vector<vector<idx_t>>();
        this->values = vector<scalar_t>();
        this->size = 0;
    }

    vector<idx_t> get_shape() {return this->shape;}
};


/***************
 ** UTILITIES **
 ***************/
void FromNumpy(py::array_t<scalar_t> a, SparseArray* out) {
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
    out->values = vector<scalar_t>(sparse_size);
    out->indices = vector<vector<idx_t>>(sparse_size, vector<idx_t>(ndim));

    size_t sparse_idx = 0;
    for (size_t k = 0; k < size; k++) {
        if (ptr[k] != out->default_val) {
            out->values[sparse_idx] = ptr[k];
            size_t idx = k;
            for (ssize_t i = ndim-1; i >= 0; i--) {
                out->indices[sparse_idx][i] = (idx % shape[i]);
                idx /= shape[i];
            }
            sparse_idx++;
        }
    }
    out->size = sparse_size;
    out->ndim = ndim;
}

vector<idx_t> compact_strides(vector<idx_t> shape) {
    vector<idx_t> strides;
    strides.reserve(shape.size());
    idx_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        strides.push_back(stride);
        stride *= shape[i];
    }
    std::reverse(strides.begin(), strides.end());
    return strides;
}

py::array_t<scalar_t> ToNumpy(const SparseArray& a) {
    vector<idx_t> shape = a.shape;
    size_t ndim = a.ndim;
    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<idx_t>());

    py::array_t<scalar_t> out = py::array_t<scalar_t>(shape);
    scalar_t* ptr = (scalar_t*)out.request().ptr;

    for (size_t i = 0; i < size; i++) {
        ptr[i] = a.default_val;
    }

    vector<idx_t> strides = compact_strides(shape);
    for (size_t i = 0; i < a.size; i++) {
        size_t idx = 0;
        for (size_t j = 0; j < ndim; j++) {
            idx += a.indices[i][j] * strides[j];
        }
        ptr[idx] = a.values[i];
    }
    return out;
}


/*********************
 ** ARRAY MANIP OPS **
 *********************/
void reshape(const SparseArray& a, vector<idx_t> new_shape, SparseArray* out) {
    out->default_val = a.default_val;
    out->shape.erase(out->shape.begin(), out->shape.end());
    out->shape.insert(out->shape.begin(), new_shape.begin(), new_shape.end());
    out->ndim = new_shape.size();
    out->values.assign(a.values.begin(), a.values.end());
    out->size = a.size;

    vector<idx_t> strides = compact_strides(a.shape);
    vector<idx_t> new_indices;
    for (int i = 0; i < a.size; i++) {
        idx_t idx = 0;
        for (int j = 0; j < a.ndim; j++) {
            idx += a.indices[i][j] * strides[j];
        }
        new_indices.push_back(idx);
    }
    vector<idx_t> new_strides = compact_strides(new_shape);
    out->indices = vector<vector<idx_t>>(out->size);
    for (int i = 0; i < out->size; i++) {
        out->indices[i] = vector<idx_t>(out->ndim);
        for (int j = 0; j < out->ndim; j++) {
            out->indices[i][j] = new_indices[i] / new_strides[j];
            new_indices[i] %= new_strides[j];
        }
    }
}

void permute(const SparseArray& a, vector<idx_t> new_axes, SparseArray* out) {
    out->default_val = a.default_val;
    out->ndim = a.ndim;
    out->indices.assign(a.indices.begin(), a.indices.end());
    out->values.assign(a.values.begin(), a.values.end());
    out->size = a.size;
    out->shape.assign(a.shape.begin(), a.shape.end());
    //permute indices according to new_axes
    for (int i = 0; i < out->size; i++) {
        for (int j = 0; j < out->ndim; j++) {
            out->indices[i][j] = a.indices[i][new_axes[j]];
            out->shape[j] = a.shape[new_axes[j]];
        }
    }
    vector<size_t> idxs(out->indices.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::stable_sort(idxs.begin(), idxs.end(),
                    [out](size_t i1, size_t i2) {
                        return out->indices[i1] < out->indices[i2];
                    });

    vector<vector<idx_t>> new_indices;
    vector<scalar_t> new_values;
    new_indices.reserve(out->indices.size());
    new_values.reserve(out->indices.size());

    for (auto i : idxs) {
        new_indices.push_back(out->indices[i]);
        new_values.push_back(out->values[i]);
    }

    out->indices = new_indices;
    out->values = new_values;
}

void broadcast_to(const SparseArray& a, const std::vector<idx_t>& new_shape, SparseArray* out) {
    out->default_val = a.default_val;
    out->shape = new_shape;
    out->ndim = new_shape.size();
    out->indices = a.indices;
    out->values = a.values;
    out->size = a.size;

    for (int ax = 0; ax < new_shape.size(); ax++) {
        int new_ax = new_shape[new_shape.size() - 1 - ax];
        int old_ax = a.shape[a.shape.size() - 1 - ax];

        if (new_ax == old_ax) {
            continue;
        } else if (old_ax == 1) {
            //copy out indices new_ax times
            //resize each ndim of out.indices to new_ax * out.size
            int og_size = out->size;
            out->indices.reserve(new_ax * og_size);
            for (int i = 1; i < new_ax; i++) {
                out->indices.insert(out->indices.end(), out->indices.begin(), out->indices.begin() + og_size);
                for (int j = 0; j < og_size; j++) {
                    out->indices[j + i*og_size][out->ndim - 1 - ax] = i;
                }
            }
            //copy out values new_ax times
            out->values.reserve(new_ax * og_size);
            for (int i = 1; i < new_ax; i++) {
                out->values.insert(out->values.end(), out->values.begin(), out->values.begin() + og_size);
            }
            out->size = out->values.size();
        } else {
            throw std::runtime_error("Cannot broadcast to shape");
        }
    }
}


/****************
 ** GETITEM OP **
 ****************/
void getitem(const SparseArray& a, const std::vector<vector<idx_t> >& slices, SparseArray* out) {
    out->default_val = a.default_val;
    for (uint i = 0; i < a.size; i++) {
        bool in_slice = true;
        for (uint j = 0; j < a.ndim; j++) {
            if (a.indices[i][j] < slices[j][0] || a.indices[i][j] >= slices[j][1] || (a.indices[i][j] - slices[j][0]) % slices[j][2] != 0) {
                in_slice = false;
                break;
            }
        }
        if (in_slice) {
            out->indices.push_back(vector<idx_t>(a.ndim));
            for(uint j=0;j<a.ndim;j++) {
                out->indices.back()[j] = (a.indices[i][j] - slices[j][0]) / slices[j][2];
            }
            out->values.push_back(a.values[i]);
        }
    }
    out->size = out->values.size();
}


/*****************
 ** SETITEM OPS **
 *****************/
void cartesian_product(vector<vector<idx_t> >& idxs, vector<vector<idx_t> >& new_indices, vector<idx_t>& temp, int idx) {
    if (idx == idxs.size()) {
        new_indices.push_back(temp);
        return;
    }
    for (int i = 0; i < idxs[idx].size(); i++) {
        temp.push_back(idxs[idx][i]);
        cartesian_product(idxs, new_indices, temp, idx + 1);
        temp.pop_back();
    }
}

/* SCALAR SETITEM */
void ScalarSetitem(SparseArray& array,const std::vector<vector<idx_t> >& slices, scalar_t val) {
    //We will assume the user knows what they are doing and will not
    //set an excessive number of elements to a new value.

    //Now take cartesian product of list1 list 2 .... in slices
    vector<vector<idx_t> > idxs;
    for (uint i = 0; i < slices.size(); i++) {
        vector<idx_t> idx;
        for (idx_t j = slices[i][0]; j < slices[i][1]; j += slices[i][2]) {
            idx.push_back(j);
        }
        idxs.push_back(idx);
    }

    //Now take cartesian product of idxs
    vector<vector<idx_t> > set_indices;
    vector<idx_t> temp;
    cartesian_product(idxs, set_indices, temp, 0);

    int a_idx = 0;
    vector<vector<idx_t> > new_indices;
    vector<scalar_t> new_values;

    for (int i = 0; i < set_indices.size(); i++) {
        while (a_idx < array.size && array.indices[a_idx] < set_indices[i]) {
            new_indices.push_back(array.indices[a_idx]);
            new_values.push_back(array.values[a_idx]);
            a_idx++;
        }
        if (a_idx < array.size && array.indices[a_idx] == set_indices[i]) {
            new_indices.push_back(set_indices[i]);
            new_values.push_back(val);
            a_idx++;
        } else {
            new_indices.push_back(set_indices[i]);
            new_values.push_back(val);
        }
    }

    while (a_idx < array.size) {
        new_indices.push_back(array.indices[a_idx]);
        new_values.push_back(array.values[a_idx]);
        a_idx++;
    }

    if (new_values.size() > 0) {
        array.indices = new_indices;
        array.values = new_values;
        array.size = new_values.size();
    } else {
        array.indices = vector<vector<idx_t> >();
        array.values = vector<scalar_t>();
        array.size = 0;
    }
}

/* EWISE SETITEM */
void EwiseSetitem(SparseArray& a,const std::vector<vector<idx_t> >& slices, SparseArray& b) {
    // Here we have to ensure the default values are the same, otherwise
    // the sparsity of the array will be destroyed.
    int a_idx = 0;
    int b_idx = 0;
    vector<vector<idx_t> > new_indices;
    vector<scalar_t> new_values;

    vector<vector<idx_t> > a_idxs;
    for (uint i = 0; i < slices.size(); i++) {
        vector<idx_t> idx;
        for (idx_t j = slices[i][0]; j < slices[i][1]; j += slices[i][2]) {
            idx.push_back(j);
        }
        a_idxs.push_back(idx);
    }

    vector<vector<idx_t> > b_idxs;
    for (uint i = 0; i < b.ndim; i++) {
        vector<idx_t> idx;
        for (idx_t j = 0; j < b.shape[i]; j++) {
            idx.push_back(j);
        }
        b_idxs.push_back(idx);
    }

    vector<vector<idx_t> > a_set_indices;
    vector<idx_t> temp;
    cartesian_product(a_idxs, a_set_indices, temp, 0);
    vector<vector<idx_t> > b_set_indices;
    temp.clear();
    cartesian_product(b_idxs, b_set_indices, temp, 0);
    for (int i = 0; i < a_set_indices.size(); i++) {
        while (a_idx < a.size && a.indices[a_idx] < a_set_indices[i]) {
            new_indices.push_back(a.indices[a_idx]);
            new_values.push_back(a.values[a_idx]);
            a_idx++;
        }
        if (a_idx < a.size && a.indices[a_idx] == a_set_indices[i]) {
            if (b_idx < b.size && b.indices[b_idx] == b_set_indices[i]) {
                new_indices.push_back(a_set_indices[i]);
                new_values.push_back(b.values[b_idx]);
                a_idx++;
                b_idx++;
            } else {
                if (b.default_val != a.default_val) {
                    new_indices.push_back(a_set_indices[i]);
                    new_values.push_back(b.default_val);
                }
                a_idx++;
            }
        }
        else{
            if (b_idx < b.size && b.indices[b_idx] == b_set_indices[i]) {
                new_indices.push_back(a_set_indices[i]);
                new_values.push_back(b.values[b_idx]);
                b_idx++;
            } else {
                throw std::runtime_error("Something went wrong");
            }
        }
    }

    while(a_idx < a.size) {
        new_indices.push_back(a.indices[a_idx]);
        new_values.push_back(a.values[a_idx]);
        a_idx++;
    }

    if(new_values.size() > 0) {
        a.indices = new_indices;
        a.values = new_values;
        a.size = new_values.size();
    } else {
        a.indices = vector<vector<idx_t> >();
        a.values = vector<scalar_t>();
        a.size = 0;
    }
}


/***************
 ** SCALAR OP **
 ***************/
void ScalarOp(const SparseArray& a, scalar_t val, SparseArray* out,
                            std::function<scalar_t(scalar_t, scalar_t)> op, bool improve_sparsity=false) {
    out->shape = a.shape;
    out->ndim = a.ndim;
    out->default_val = op(a.default_val, val);

    out->values.reserve(a.size);
    out->indices.reserve(a.size);

    scalar_t new_val;
    for (size_t i = 0; i < a.size; i++) {
        new_val = op(a.values[i], val);
        if (!improve_sparsity || (new_val != out->default_val)) {
            out->values.push_back(new_val);
            out->indices.push_back(a.indices[i]);
        }
    }
    out->size = out->values.size();
}


/********************
 ** SINGLE ARR OPS **
 ********************/
void EwiseNeg(const SparseArray& a, SparseArray* out) {
    ScalarOp(a, 0, out, [](scalar_t a, scalar_t b) { return -a; });
}

void EwiseLog(const SparseArray& a, SparseArray* out) {
    ScalarOp(a, 0, out, [](scalar_t a, scalar_t b) { return std::log(a); });
}

void EwiseExp(const SparseArray& a, SparseArray* out) {
    ScalarOp(a, 0, out, [](scalar_t a, scalar_t b) { return std::exp(a); });
}

void EwiseTanh(const SparseArray& a, SparseArray* out) {
    ScalarOp(a, 0, out, [](scalar_t a, scalar_t b) { return std::tanh(a); });
}


/**********************
 ** SCALAR ARITH OPS **
 **********************/
void ScalarAdd(const SparseArray& a, scalar_t val, SparseArray* out) {
    ScalarOp(a, val, out, std::plus<scalar_t>());
}

void ScalarSub(const SparseArray& a, scalar_t val, SparseArray* out) {
    ScalarOp(a, val, out, std::minus<scalar_t>());
}

void ScalarMul(const SparseArray& a, scalar_t val, SparseArray* out) {
    ScalarOp(a, val, out, std::multiplies<scalar_t>());
}

void ScalarDiv(const SparseArray& a, scalar_t val, SparseArray* out) {
    ScalarOp(a, val, out, std::divides<scalar_t>());
}

void ScalarPower(const SparseArray& a, scalar_t val, SparseArray* out) {
    ScalarOp(a, val, out, [](scalar_t a, scalar_t b) { return std::pow(a, b); });
}


/*********************
 ** SCALAR COMP OPS **
 *********************/
void ScalarMaximum(const SparseArray& a, scalar_t val, SparseArray* out) {
    ScalarOp(a, val, out, [](scalar_t a, scalar_t b) { return  std::max(a, b);  }, true);
}

void ScalarEq(const SparseArray& a, scalar_t val, SparseArray* out) {
    ScalarOp(a, val, out, [](scalar_t a, scalar_t b) { return (a == b) ? 1 : 0; }, true);
}

void ScalarNe(const SparseArray& a, scalar_t val, SparseArray* out) {
    ScalarOp(a, val, out, [](scalar_t a, scalar_t b) { return (a != b) ? 1 : 0; }, true);
}

void ScalarGe(const SparseArray& a, scalar_t val, SparseArray* out) {
    ScalarOp(a, val, out, [](scalar_t a, scalar_t b) { return (a >= b) ? 1 : 0; }, true);
}

void ScalarLe(const SparseArray& a, scalar_t val, SparseArray* out) {
    ScalarOp(a, val, out, [](scalar_t a, scalar_t b) { return (a <= b) ? 1 : 0; }, true);
}

void ScalarGt(const SparseArray& a, scalar_t val, SparseArray* out) {
    ScalarOp(a, val, out, [](scalar_t a, scalar_t b) { return (a > b) ? 1 : 0; }, true);
}

void ScalarLt(const SparseArray& a, scalar_t val, SparseArray* out) {
    ScalarOp(a, val, out, [](scalar_t a, scalar_t b) { return (a < b) ? 1 : 0; }, true);
}


/**************
 ** EWISE OP **
 **************/
void EwiseOp(const SparseArray a, const SparseArray b, SparseArray* out,
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
void EwiseAdd(const SparseArray& a, const SparseArray& b, SparseArray* out) {
    EwiseOp(a, b, out, std::plus<scalar_t>(),
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}

void EwiseSub(const SparseArray& a, const SparseArray& b, SparseArray* out) {
    EwiseOp(a, b, out, std::minus<scalar_t>(),
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}

void EwiseMul(const SparseArray& a, const SparseArray& b, SparseArray* out) {
    EwiseOp(a, b, out, std::multiplies<scalar_t>(),
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}

void EwiseDiv(const SparseArray& a, const SparseArray& b, SparseArray* out) {
    EwiseOp(a, b, out, std::divides<scalar_t>(),
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}


/********************
 ** EWISE COMP OPS **
 ********************/
void EwiseMaximum(const SparseArray& a, const SparseArray& b, SparseArray* out) {
    EwiseOp(a, b, out, [](scalar_t a, scalar_t b) { return std::max(a, b); },
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}

void EwiseEq(const SparseArray& a, const SparseArray& b, SparseArray* out) {
    EwiseOp(a, b, out, [](scalar_t a, scalar_t b) { return (a == b) ? 1 : 0; },
                    [](size_t a_size, size_t b_size) { return std::min(a_size, b_size); });
}

void EwiseNe(const SparseArray& a, const SparseArray& b, SparseArray* out) {
    EwiseOp(a, b, out, [](scalar_t a, scalar_t b) { return (a != b) ? 1 : 0; },
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}

void EwiseGe(const SparseArray& a, const SparseArray& b, SparseArray* out) {
    EwiseOp(a, b, out, [](scalar_t a, scalar_t b) { return (a >= b) ? 1 : 0; },
                    [](size_t a_size, size_t b_size) { return std::min(a_size, b_size); });
}

void EwiseLe(const SparseArray& a, const SparseArray& b, SparseArray* out) {
    EwiseOp(a, b, out, [](scalar_t a, scalar_t b) { return (a <= b) ? 1 : 0; },
                    [](size_t a_size, size_t b_size) { return std::min(a_size, b_size); });
}

void EwiseGt(const SparseArray& a, const SparseArray& b, SparseArray* out) {
    EwiseOp(a, b, out, [](scalar_t a, scalar_t b) { return (a > b) ? 1 : 0; },
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}

void EwiseLt(const SparseArray& a, const SparseArray& b, SparseArray* out) {
    EwiseOp(a, b, out, [](scalar_t a, scalar_t b) { return (a < b) ? 1 : 0; },
                    [](size_t a_size, size_t b_size) { return a_size + b_size; });
}


/***************
 ** MATMUL OP **
 ***************/
void matmul(const SparseArray& a, const SparseArray& b, SparseArray& out) {
    assert(a.shape[1] == b.shape[0]);
    out.ndim = 2;
    out.default_val = a.default_val * b.default_val * a.shape.back();
    std::vector<float> a_devs(a.values.size());
    std::vector<float> b_devs(b.values.size());

    for (uint i = 0; i < a.values.size(); i++) {
        a_devs[i] = a.values[i] - a.default_val;
    }

    for (uint i = 0; i < b.values.size(); i++) {
        b_devs[i] = b.values[i] - b.default_val;
    }

    std::unordered_map<uint, std::deque<std::pair<uint, float>>> dict_a_ind_1;
    std::unordered_map<uint, std::deque<std::pair<uint, float>>> dict_b_ind_0;

    for (uint i = 0; i < a.size; i++) {
        dict_a_ind_1[a.indices[i][1]].push_back({a.indices[i][0], a_devs[i]});
    }

    for (uint i = 0; i < b.size; i++) {
        dict_b_ind_0[b.indices[i][0]].push_back({b.indices[i][1], b_devs[i]});
    }

    std::map<std::pair<uint, uint>, float> out_dict;

    for (const auto& [idx, a_list] : dict_a_ind_1) {
        const auto& b_list = dict_b_ind_0[idx];
        for (const auto& [i, a_val] : a_list) {
            for (const auto& [j, b_val] : b_list) {
                //check if key exists
                if (out_dict.find({i, j}) == out_dict.end()) out_dict[{i, j}] = 0;
                out_dict[{i, j}] += a_val * b_val;
            }
        }
    }

    if (a.default_val != 0) {
        for (uint i = 0; i < a.size; i++) {
            const auto& idx = a.indices[i];
            const auto& val = a_devs[i];
            for (uint j = 0; j < b.shape[1]; j++) {
                out_dict[{idx[0], j}] += val * b.default_val;
            }
        }
    }

    if (b.default_val != 0) {
        for (uint i = 0; i < b.size; i++) {
            const auto& idx = b.indices[i];
            const auto& val = b_devs[i];
            for (uint j = 0; j < a.shape[0]; j++) {
                out_dict[{j, idx[1]}] += a.default_val * val;
            }
        }
    }

    for (const auto& [idx, val] : out_dict) {
        out.indices.push_back(std::vector<idx_t>{idx.first, idx.second});
        out.values.push_back(val + out.default_val);
    }

    out.size = out.values.size();
}



/*******************
 ** REDUCTION OPS **
 *******************/
void reduce_max(const SparseArray& a, uint axis, bool keepdims, SparseArray* out) {
    vector<idx_t> reduced_shape = a.shape;
    reduced_shape.erase(reduced_shape.begin() + axis);

    vector<idx_t> reduced_strides = compact_strides(reduced_shape);
    vector<vector<idx_t>> reduced_indices = a.indices;
    std::transform(reduced_indices.begin(), reduced_indices.end(), reduced_indices.begin(),
                   [axis](vector<idx_t> idx) { idx.erase(idx.begin() + axis); return idx; });

    vector<size_t> idxs(reduced_indices.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::stable_sort(idxs.begin(), idxs.end(),
                    [&reduced_indices](size_t i1, size_t i2) {
                        return reduced_indices[i1] < reduced_indices[i2];
                    });

    vector<vector<idx_t>> new_indices;
    vector<scalar_t> new_values;
    new_indices.reserve(reduced_indices.size());
    new_values.reserve(reduced_indices.size());

    size_t i = 0;
    size_t counter;
    do {
        vector<idx_t> new_idxs = reduced_indices[idxs[i]];
        if (keepdims) { new_idxs.insert(new_idxs.begin() + axis, 0); }
        new_indices.push_back(new_idxs);
        new_values.push_back(a.values[idxs[i]]);
        i++; counter = 1;

        while (i < idxs.size() && reduced_indices[idxs[i]] == reduced_indices[idxs[i-1]]) {
            new_values.back() = std::max(new_values.back(), a.values[idxs[i]]);
            i++; counter++;
        }
        if (counter < a.shape[axis]) {
            new_values.back() = std::max(new_values.back(), a.default_val);
        }

    } while (i < idxs.size());

    out->default_val = a.default_val;
    out->indices = new_indices;
    out->values = new_values;
    out->size = new_values.size();
}

void reduce_sum(const SparseArray& a, uint axis, bool keepdims, SparseArray* out) {
    vector<idx_t> reduced_shape = a.shape;
    reduced_shape.erase(reduced_shape.begin() + axis);

    vector<idx_t> reduced_strides = compact_strides(reduced_shape);
    vector<vector<idx_t>> reduced_indices = a.indices;
    std::transform(reduced_indices.begin(), reduced_indices.end(), reduced_indices.begin(),
                   [axis](vector<idx_t> idx) { idx.erase(idx.begin() + axis); return idx; });

    vector<size_t> idxs(reduced_indices.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::stable_sort(idxs.begin(), idxs.end(),
                    [&reduced_indices](size_t i1, size_t i2) {
                        return reduced_indices[i1] < reduced_indices[i2];
                    });

    vector<vector<idx_t>> new_indices;
    vector<scalar_t> new_values;
    new_indices.reserve(reduced_indices.size());
    new_values.reserve(reduced_indices.size());

    size_t i = 0;
    size_t counter;
    do {
        vector<idx_t> new_idxs = reduced_indices[idxs[i]];
        if (keepdims) { new_idxs.insert(new_idxs.begin() + axis, 0); }
        new_indices.push_back(new_idxs);
        new_values.push_back(a.values[idxs[i]]);
        i++; counter = 1;

        while (i < idxs.size() && reduced_indices[idxs[i]] == reduced_indices[idxs[i-1]]) {
            new_values.back() += a.values[idxs[i]];
            i++; counter++;
        }
        new_values.back() += a.default_val * (a.shape[axis] - counter);

    } while (i < idxs.size());

    out->default_val = a.default_val * a.shape[axis];
    out->indices = new_indices;
    out->values = new_values;
    out->size = new_values.size();
}

}  // namespace cpu
}  // namespace SparseNDArray


/*****************
 ** PYBIND CODE **
 *****************/
PYBIND11_MODULE(sparse_ndarray_backend_cpu, m) {
    using namespace SparseNDArray;
    using namespace cpu;

    m.attr("__device_name__") = "cpu";

    py::class_<SparseArray>(m, "SparseArray")
            .def(py::init<vector<idx_t>, scalar_t>(), py::return_value_policy::take_ownership)
            .def("fill", &SparseArray::Fill)
            .def("get_shape", &SparseArray::get_shape)
            .def_readonly("size", &SparseArray::size)
            .def_readonly("ndim", &SparseArray::ndim)
            .def_readwrite("default_val", &SparseArray::default_val);

    // UTILITIES
    m.def("from_numpy", FromNumpy);
    m.def("to_numpy", ToNumpy);

    // ARRAY MANIP OPS
    m.def("reshape", reshape);
    m.def("permute", permute);
    m.def("broadcast_to", broadcast_to);

    // GETITEM SETITEM OPS
    m.def("getitem", getitem);
    m.def("ewise_setitem", EwiseSetitem);
    m.def("scalar_setitem", ScalarSetitem);

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

    // MATMUL OP
    m.def("matmul", matmul);

    // REDUCTION OPS
    m.def("reduce_max", reduce_max);
    m.def("reduce_sum", reduce_sum);
}
