from typing import Callable
from collections import deque, defaultdict

import numpy as np
import itertools

__device_name__ = "numpy"
_datatype = np.float32


####################
### SPARSE ARRAY ###
####################
class SparseArray:
    def __init__(self,
                 shape: tuple[int, ...],
                 default_val: float = 0,
                 indices: np.ndarray = None,
                 values: np.ndarray = None):
        self.shape = shape
        self.ndim = len(shape)
        self.default_val = default_val

        # If no indices are provided we initialize indices to an empty array
        if indices is None:
            self.indices = np.empty((0, self.ndim), dtype=np.uint32)
        else:
            self.indices = indices

        # If no values are provided we initialize values to an empty array
        if values is None:
            self.values = np.empty((0,), dtype=_datatype)
        else:
            self.values = values

        # Check that the provided indices and values are valid
        SparseArray.check(shape, self.indices, self.values)

        # Compute the number of non-default values in the sparse array
        self.size = self.values.size

    def fill(self, value):
        # Fill overwrites all values with the provided value so
        # it becomes the new default value
        self.default_val = value
        # We can simply empty the indices and values arrays
        self.indices = np.empty((0, self.ndim), dtype=np.uint32)
        self.values = np.empty((0,), dtype=_datatype)
        # No non-default values remain
        self.size = 0

    def get_shape(self):
        return self.shape

    @staticmethod
    def check(shape, indices, values):
        assert isinstance(shape, tuple)
        assert indices.dtype == np.uint32
        assert values.dtype == _datatype
        assert values.ndim == 1 and indices.ndim == 2 # 1D values, 2D indices
        assert indices.shape[0] == values.shape[0] # same number of values and indices
        assert indices.shape[1] == len(shape) # indices have same number of dimensions as shape


#################
### UTILITIES ###
#################
def from_numpy(other: np.ndarray, out: SparseArray):
    # find the most common value in the array to use as the default value
    unique, counts = np.unique(other, return_counts=True)
    unique = unique.astype(np.float32)
    max_count_i = np.argmax(counts)
    default_val = unique[max_count_i]

    sparse_size = other.size - counts[max_count_i]
    # This is a strict check that the sparse array memory usage is less than
    # the dense array. However, we can relax this check for now.
    # if (sparse_size * (other.ndim + 1) > other.size):
    #     raise ValueError("Sparse array memory usage is greater than dense array")

    values = [0.0] * sparse_size
    indices = [None] * sparse_size
    # find all non-default values and their indices
    if sparse_size > 0:
        i = 0
        for index, value in np.ndenumerate(other):
            if value != default_val:
                indices[i] = index
                values[i] = value
                i += 1

    shape = other.shape
    values = np.array(values, dtype=_datatype)
    indices = np.array(indices, dtype=np.uint32).reshape((len(values), len(shape)))
    SparseArray.check(shape, indices, values)

    out.shape = shape
    out.ndim = len(shape)
    out.default_val = default_val
    out.indices = indices
    out.values = values
    out.size = values.size


def compact_strides(shape: tuple[int, ...]):
        """Utility function to compute compact strides"""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])


def to_numpy(a: SparseArray):
    # Create a full array with the default value
    out = np.full(a.shape, a.default_val, dtype=np.float32)
    # Retrieve all non-default values and their indices
    for idx, val in zip(a.indices, a.values):
        out[tuple(idx)] = val
    return out


##############################
### ARRAY MANIPULATION OPS ###
##############################
def reshape(a: SparseArray, new_shape: tuple[int, ...], out: SparseArray):
    out.default_val = a.default_val
    out.shape = new_shape
    out.ndim = len(new_shape)
    out.values = a.values.copy()
    out.size = a.size
    # Use compact strides to compute the global index of each non-default-val value
    out.indices = np.sum(a.indices * compact_strides(a.shape), axis=1)
    # Unravel the global indices according to the new output shape
    out.indices = np.array(np.unravel_index(out.indices, new_shape)).T
    out.indices = out.indices.astype(np.uint32)


def permute(a: SparseArray, new_axes: tuple[int, ...], out: SparseArray):
    out.default_val = a.default_val
    out.ndim = len(out.shape)
    # Permute the axes by directly permuting the ndim indices
    out.indices = a.indices[:, new_axes]
    out.values = a.values.copy()
    # Preserve the invariant that indices are sorted by global index
    order = np.argsort((out.indices * compact_strides(out.shape)).sum(axis=1))
    # Sort both indices and values by the global index
    out.indices = out.indices[order]
    out.values = out.values[order]
    out.size = out.values.size


def broadcast_to(a: SparseArray, new_shape: tuple[int, ...], out: SparseArray):
    out.default_val = a.default_val
    out.shape = new_shape
    out.ndim = len(new_shape)
    out.indices = a.indices.copy()
    out.values = a.values.copy()
    out.size = a.size

    # Iterate backwards over the ndims to support broadcasting over multiple axes
    for ax, (new_ax, old_ax) in enumerate(zip(new_shape[::-1], a.shape[::-1])):
        if new_ax == old_ax:
            continue
        elif old_ax == 1:
            # Repeat the indices new_ax times along axis=1 to repeat ndim indices
            out.indices = np.tile(out.indices, (new_ax, 1))
            # Correct the index at the broadcasted axis in the repeated indices
            for i in range(out.size):
                out.indices[i*out.size:(i+1)*out.size, out.ndim-1-ax] = i
            # Similarly repeat the values new_ax times
            out.values = np.tile(out.values, new_ax)
            out.size = out.values.size
        else:
            raise RuntimeError("Cannot broadcast to shape")


##################
### GETITEM OP ###
##################
def getitem(a: SparseArray, idxs: tuple[tuple, ...], out: SparseArray):

    new_indices, new_values = a.indices.copy(), a.values.copy()

    # Iterate through each index in a.indices and check whether it lies in
    # the slice given by idxs. If not, mask it out.
    mask = np.zeros_like(new_indices, dtype=np.bool_)
    for s_ind, s in enumerate(idxs):
        mask[:, s_ind] = (new_indices[:, s_ind][:, None] ==
                    np.arange(s[0], s[1], s[2])[None, :]).any(axis=1)
        new_indices[:, s_ind] = (new_indices[:, s_ind] - s[0]) / s[2]

    mask = mask.all(axis=1)
    new_indices = new_indices[mask]
    new_values = new_values[mask]

    out.default_val = a.default_val
    out.indices = new_indices
    out.values = new_values
    out.size = out.values.size


#########################
### SCALAR SETITEM OP ###
#########################
def scalar_setitem(a: SparseArray, idxs: tuple[tuple, ...], val: float):
    # If the provided val is the default value, we can simply remove the
    # indices that are being set to the default value from the sparse array.
    if val == a.default_val:
        mask = np.zeros_like(a.indices, dtype=np.bool_)
        for s_ind, s in enumerate(idxs):
            mask[:, s_ind] = (a.indices[:, s_ind][:, None] ==
                        np.arange(s[0], s[1], s[2])[None, :]).any(axis=1)

        mask = mask.all(axis=1)
        a.indices = a.indices[~mask]
        a.values = a.values[~mask]
        return

    a_idx = 0   # SparseArray index into a.indices and a.values
    new_indices = deque()
    new_values = deque()

    # We iterate through all indices being set (given in idxs) using the following
    # itertools utility function
    for i in itertools.product(*[list(range(s[0], s[1], s[2])) for s in idxs]):
        # We iterate through all indices in the sparse array that are less than
        # the current index being set (we know the sparse array indices are sorted)
        while a_idx < a.size and tuple(a.indices[a_idx]) < i:
            # Since we are not setting any of these indices, we can simply copy
            # them over to the new sparse array
            new_indices.append(tuple(a.indices[a_idx]))
            new_values.append(a.values[a_idx])
            a_idx += 1

        # If the current index being set is already in the sparse array, we
        # simply update the value at that index
        if a_idx < a.size and tuple(a.indices[a_idx]) == i:
            new_indices.append(i)
            new_values.append(val)
            a_idx += 1  # We increment a_idx to skip over the index we just set
        else:
            # If the current index being set is not in the sparse array (guaranteed,
            # by the fact that we iterated through all indices less than it, and are now
            # at an index greater than it), we add it to the new sparse array
            new_indices.append(i)
            new_values.append(val)

    # Finally, we copy over any remaining indices from the sparse array
    while a_idx < a.size:
        new_indices.append(tuple(a.indices[a_idx]))
        new_values.append(a.values[a_idx])
        a_idx += 1

    # If the new sparse array is non-empty, we copy it over to the original sparse array
    if new_values:
        a.indices = np.array(new_indices, dtype=np.uint32)
        a.values = np.array(new_values, dtype=_datatype)
        a.size = a.values.size
    else:
        # If the new sparse array is empty, we simply set the original sparse array
        a.indices = np.empty((0, a.ndim), dtype=np.uint32)
        a.values = np.empty((0,), dtype=_datatype)
        a.size = 0


########################
### EWISE SETITEM OP ###
########################
def ewise_setitem(a: SparseArray, idxs: tuple[tuple, ...], b: SparseArray):

    a_idx = 0   # SparseArray index into a.indices and a.values
    b_idx = 0   # SparseArray index into b.indices and b.values
    new_indices = deque()
    new_values = deque()

    idxs_a = [list(range(s[0], s[1], s[2])) for s in idxs]  # dense array indices into a
    idxs_b = [list(range(0, dim)) for dim in b.shape]       # dense array indices into b

    for i_a, i_b in zip(itertools.product(*idxs_a), itertools.product(*idxs_b)):
        # We iterate through all indices in a that are less than the current index
        # being set (we know the sparse array indices are sorted)
        while a_idx < a.size and tuple(a.indices[a_idx]) < i_a:
            # Since we are not setting any of these indices, we can simply copy
            # them over to the new sparse array
            new_indices.append(tuple(a.indices[a_idx]))
            new_values.append(a.values[a_idx])
            a_idx += 1

        # If the current index being set is already in the sparse array, we
        # simply update the value at that index
        if a_idx < a.size and tuple(a.indices[a_idx]) == i_a:
            # Unlike before we need to check whether the value we are setting
            # is a non-default value in b by comparing i_b to b.indices[b_idx]
            if b_idx < b.size and i_b == tuple(b.indices[b_idx]):
                # If the value is non-default, we set the existing non-default
                # value in a to the non-default value in b
                new_indices.append(i_a)
                new_values.append(b.values[b_idx])
                # increment b_idx here because we have consumed the value at this index
                b_idx += 1
            else:
                # If the value is default, we set the existing non-default value
                # in a to the default value from b (only if they are not equal)
                if b.default_val != a.default_val:
                    new_indices.append(i_a)
                    new_values.append(b.default_val)
            # always increment our sparse index into a here because we have set
            # the value at this index
            a_idx += 1
        else:
            # If the current index being set is not in the sparse array a (guaranteed,
            # by the fact that we iterated through all indices less than it, and are now
            # at an index greater than it), we add it to the new sparse array. But again,
            # we need to check whether the value we are setting is a non-default value in b
            if b_idx < b.size and i_b == tuple(b.indices[b_idx]):
                # If the value is non-default, we create a new entry in a with the current
                # dense index into a and the non-default value from b
                new_indices.append(i_a)
                new_values.append(b.values[b_idx])
                # increment b_idx here because we have consumed the value at this index
                b_idx += 1
            else:
                # If the value is default, we create a new entry in a with the current
                # dense index into a and the default value from b (only if they are not equal)
                if b.default_val != a.default_val:
                    new_indices.append(i_a)
                    new_values.append(b.default_val)

    # Because the size of b corresponds exactly with the idxs provided to setitem, we must have
    # consumed all of the values in b, therefore all that's left is to check if any elements in
    # a remain, and if so copy them over to the new sparse array
    while a_idx < a.size:
        new_indices.append(tuple(a.indices[a_idx]))
        new_values.append(a.values[a_idx])
        a_idx += 1

    # If the new sparse array is non-empty, we copy it over to the original sparse array
    if new_values:
        a.indices = np.array(new_indices, dtype=np.uint32)
        a.values = np.array(new_values, dtype=_datatype)
        a.size = a.values.size
    else:
        # If the new sparse array is empty, we simply set the original sparse array
        a.indices = np.empty((0, a.ndim), dtype=np.uint32)
        a.values = np.empty((0,), dtype=_datatype)
        a.size = 0


#################
### SCALAR OP ###
#################
def scalar_op(a: SparseArray, val: float, out: SparseArray, op: Callable, sparsify: bool = True):
    # Apply the operation to the default value
    out.default_val = op(a.default_val, val)
    out.shape = a.shape
    out.ndim = a.ndim

    out.indices = a.indices.copy()
    # Apply the scalar operation to all non-default values
    out.values = op(a.values, val).astype(_datatype)
    out.size = out.values.size

    if sparsify:
        # Remove indices where the value is now the default value
        mask = out.values != out.default_val
        out.indices = out.indices[mask]
        out.values = out.values[mask]
        out.size = out.values.size


######################
### SINGLE ARR OPS ###
######################
def ewise_neg(a: SparseArray, out: SparseArray):
    scalar_op(a, 0, out, lambda x, _: -x)

def ewise_log(a: SparseArray, out: SparseArray):
    scalar_op(a, 0, out, lambda x, _: np.log(x))

def ewise_exp(a: SparseArray, out: SparseArray):
    scalar_op(a, 0, out, lambda x, _: np.exp(x))

def ewise_tanh(a: SparseArray, out: SparseArray):
    scalar_op(a, 0, out, lambda x, _: np.tanh(x))


########################
### SCALAR ARITH OPS ###
########################
def scalar_add(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: x + y)

def scalar_sub(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: x - y)

def scalar_mul(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: x * y)

def scalar_div(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: x / y)

def scalar_power(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: x**y)


#######################
### SCALAR COMP OPS ###
#######################
def scalar_maximum(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: np.maximum(x, y))

def scalar_eq(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: (x == y))

def scalar_ne(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: (x != y))

def scalar_ge(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: (x >= y))

def scalar_le(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: (x <= y))

def scalar_gt(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: (x > y))

def scalar_lt(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: (x < y))


################
### EWISE OP ###
################
def ewise_op(a: SparseArray, b: SparseArray, out: SparseArray, op: Callable, sparsify: bool = True):
    assert a.shape == b.shape
    assert a.ndim == b.ndim

    out.shape = a.shape
    out.ndim = a.ndim
    # Apply the operation to the default values to get the new default value
    out.default_val = op(a.default_val, b.default_val)

    a_idx = 0   # SparseArray index into a.indices and a.values
    b_idx = 0   # SparseArray index into b.indices and b.values
    new_indices = deque()
    new_values = deque()

    while a_idx < a.size and b_idx < b.size:
        # We must convert both indices to tuples to compare them easily
        # as tuple comparisons are lexicographic from left to right
        a_i = tuple(a.indices[a_idx])
        b_i = tuple(b.indices[b_idx])

        if a_i == b_i:
            # If the indices are equal, we apply the operation to the values
            new_val = op(a.values[a_idx], b.values[b_idx])
            if not sparsify or new_val != out.default_val:
                # If the result is non-default, we add it to the new sparse array
                new_indices.append(a_i)
                new_values.append(new_val)
            # We increment both indices because we have consumed the values at both indices
            a_idx += 1
            b_idx += 1

        elif a_i <= b_i:
            # If the index from a is less than or equal to the index from b, we apply
            # the operation to the value from a and the default value from b
            new_val = op(a.values[a_idx], b.default_val)
            if not sparsify or new_val != out.default_val:
                # If the result is non-default, we add it to the new sparse array
                new_indices.append(a_i)
                new_values.append(new_val)
            # We increment only a_idx because we have consumed the value at this index
            a_idx += 1

        elif a_i >= b_i:
            # If the index from b is less than or equal to the index from a, we apply
            # the operation to the default value from a and the value from b
            new_val = op(a.default_val, b.values[b_idx])
            if not sparsify or new_val != out.default_val:
                # If the result is non-default, we add it to the new sparse array
                new_indices.append(b_i)
                new_values.append(new_val)
            # We increment only b_idx because we have consumed the value at this index
            b_idx += 1

        else:
            raise RuntimeError("Should never get here")

    while a_idx < a.size:
        a_i = tuple(a.indices[a_idx])
        new_val = op(a.values[a_idx], b.default_val)
        if not sparsify or new_val != out.default_val:
            new_indices.append(a_i)
            new_values.append(new_val)
        a_idx += 1

    while b_idx < b.size:
        b_i = tuple(b.indices[b_idx])
        new_val = op(a.default_val, b.values[b_idx])
        if not sparsify or new_val != out.default_val:
            new_indices.append(b_i)
            new_values.append(new_val)
        b_idx += 1

    if new_values:
        out.indices = np.array(new_indices, dtype=np.uint32)
        out.values = np.array(new_values, dtype=_datatype)
        out.size = out.values.size
    else:
        out.indices = np.empty((0, a.ndim), dtype=np.uint32)
        out.values = np.empty((0,), dtype=_datatype)
        out.size = 0


#######################
### EWISE ARITH OPS ###
#######################
def ewise_add(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: x + y)

def ewise_sub(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: x - y)

def ewise_mul(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: x * y)

def ewise_div(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: x / y)


######################
### EWISE COMP OPS ###
######################
def ewise_maximum(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: np.maximum(x, y))

def ewise_eq(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: (x == y))

def ewise_ne(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: (x != y))

def ewise_ge(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: (x >= y))

def ewise_le(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: (x <= y))

def ewise_gt(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: (x > y))

def ewise_lt(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: (x < y))


##############
### MATMUL ###
##############
def matmul(a: SparseArray, b: SparseArray, out: SparseArray):
    # Note that if a = a_0 + a.default_val and b = b_0 + b.default_val
    # then a @ b =  a_0 @ b_0 +                     # sparse @ sparse
    #               a_0 @ b.default_val +           # sparse @ dense (but dense is full with one value)
    #               a.default_val @ b_0 +           # dense @ sparse (but dense is full with one value)
    #               a.default_val @ b.default_val   # dense @ dense  (but both full with one value)
    assert a.shape[-1] == b.shape[-2]
    out.shape = (a.shape[0],b.shape[1])
    out.ndim = 2
    # a.default_val @ b.default_val
    out.default_val = a.default_val * b.default_val * a.shape[-1]

    a_0 = a.values - a.default_val   # a_0
    b_0 = b.values - b.default_val   # b_0

    dict_a_ind_1 = defaultdict(lambda : [])
    dict_b_ind_0 = defaultdict(lambda : [])
    for i in range(a.size):
        # Construct a dictionary mapping column index of a to (row index, value) pairs
        dict_a_ind_1[a.indices[i][1]].append((a.indices[i][0], a_0[i]))
    for i in range(b.size):
        # Construct a dictionary mapping row index of b to (column index, value) pairs
        dict_b_ind_0[b.indices[i][0]].append((b.indices[i][1], b_0[i]))

    # out_dict : dict of (i,j) -> value
    out_dict = defaultdict(lambda : 0)
    # for each key in dict_a_ind_1 and matching key in dict_b_ind_0
    # multiply values and add to out_dict
    for idx,a_list in dict_a_ind_1.items():
        # idx is the column index of a, so find matching row index of b
        b_list = dict_b_ind_0[idx]
        for i,a_val in a_list:
            for j,b_val in b_list:
                # only these outputs of sparse @ sparse will be non-zero
                # because the col_idx of a must match the row_idx of b
                # for a non-zero output (otherwise one of the values is 0)
                out_dict[(i,j)] += a_val * b_val

    # a_0 @ b.default_val
    if b.default_val != 0:
        # we only need to do this if b.default_val != 0
        for idx,val in zip(a.indices, a_0):
            for i in range(b.shape[1]):
                out_dict[(idx[0],i)] += val * b.default_val

    # a.default_val @ b_0
    if a.default_val != 0:
        # we only need to do this if a.default_val != 0
        for idx,val in zip(b.indices, b_0):
            for i in range(a.shape[0]):
                out_dict[(i,idx[1])] += a.default_val * val

    ## convert out_dict to out.indices and out.values
    out_indices = deque([])
    out_values = deque([])
    for idx,val in sorted(out_dict.items()):
        # Have to preserve the invariant that indices are sorted by global index
        out_indices.append(idx)
        out_values.append(val + out.default_val)
    out_indices = np.array(out_indices,dtype=np.uint32)
    out_values = np.array(out_values,dtype=_datatype)
    out.indices = out_indices
    out.values = out_values
    out.size = out_values.size


##################
### REDUCE OPS ###
##################
def reduce_sum(a: SparseArray, axis: int, keepdims: bool, out: SparseArray):
    # ignore keepdims for now and calculate the reduced shape
    reduced_shape = tuple([s for i, s in enumerate(a.shape) if i != axis])
    # get compact strides for reduced shape
    reduced_strides = compact_strides(reduced_shape)
    # delete the axis being reduced over from a.indices and compute the global index
    reduced_indices = np.delete(a.indices, axis, axis=1)
    # get the correct order of indices to preserve the invariant that indices are sorted
    order = np.argsort((reduced_indices * reduced_strides).sum(axis=1))

    # np.unique returns the unique elements, the index at which each unique element first
    # appears, and the number of times each unique element appears. Given we have a sorted
    # array, this is equivalent to a run-length encoding of the array. Also note that the
    # unique elements are returned in sorted order.
    unique, split_inds, counts = np.unique(reduced_indices[order], axis=0,
                                           return_index=True, return_counts=True)
    if keepdims:
        # insert the axis being reduced over back into the indices
        out.indices = np.insert(unique, axis, 0, axis=1)
    else:
        # otherwise just keep the reduced indices
        out.indices = unique

    # split the values into chunks corresponding to the run-length encoding
    splits = np.split(a.values[order], split_inds[1:])
    # pad the splits with the default value to the correct length of the original axis
    splits = [np.pad(s, (0, a.shape[axis] - counts[i]), constant_values=a.default_val) for i, s in enumerate(splits)]
    # sum the splits to get the reduced values
    out.values = np.sum(splits, axis=1)

    out.size = out.values.size
    out.default_val = a.default_val * a.shape[axis]


def reduce_max(a: SparseArray, axis: int, keepdims: bool, out: SparseArray):
    # ignore keepdims for now and calculate the reduced shape
    reduced_shape = tuple([s for i, s in enumerate(a.shape) if i != axis])
    # get compact strides for reduced shape
    reduced_strides = compact_strides(reduced_shape)
    # delete the axis being reduced over from a.indices and compute the global index
    reduced_indices = np.delete(a.indices, axis, axis=1)
    # get the correct order of indices to preserve the invariant that indices are sorted
    order = np.argsort((reduced_indices * reduced_strides).sum(axis=1))

    # np.unique returns the unique elements, the index at which each unique element first
    # appears, and the number of times each unique element appears. Given we have a sorted
    # array, this is equivalent to a run-length encoding of the array. Also note that the
    # unique elements are returned in sorted order.
    unique, split_inds, counts = np.unique(reduced_indices[order], axis=0,
                                           return_index=True, return_counts=True)
    if keepdims:
        # insert the axis being reduced over back into the indices
        out.indices = np.insert(unique, axis, 0, axis=1)
    else:
        # otherwise just keep the reduced indices
        out.indices = unique

    # split the values into chunks corresponding to the run-length encoding
    splits = np.split(a.values[order], split_inds[1:])
    # pad the splits with the default value to the correct length of the original axis
    splits = [np.pad(s, (0, a.shape[axis] - counts[i]), constant_values=a.default_val) for i, s in enumerate(splits)]
    # max the splits to get the reduced values
    out.values = np.max(splits, axis=1)

    out.size = out.values.size
    out.default_val = a.default_val

